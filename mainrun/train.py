# import utils
import math, random, time
from dataclasses import dataclass, field
import json
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog
from helper import plot_model_highlevel, profile_tokenizer, diagnose_tokenizer
from torch.utils.tensorboard import SummaryWriter
import optuna
import yaml
from typing import Any
from sam import SAM


@dataclass
class Hyperparameters:
    block_size: int = 64
    batch_size: int = 64
    vocab_size: int = 4_000
    n_layer: int = 4
    n_head: int = 3
    d_model: int = 192
    dropout: float = 0.1
    lr: float = 6e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    evals_per_epoch: int = 3
    warmup_ratio: float = 0.1

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = open(log_file, "w")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(
                        f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s"
                    )
                else:
                    parts = [
                        f"{k}={v}"
                        for k, v in kwargs.items()
                        if k not in ["prnt", "timestamp"]
                    ]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)


logger = None


def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset(
        "julien040/hacker-news-posts", split="train", cache_dir="./data"
    ).shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


def get_batch(
    split_ids: torch.Tensor,
    ptr: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr : ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


def iter_full_split(
    split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device
):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr : ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        # self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        # self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    def save(self, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.tk.save(save_path)

    @classmethod
    def load(cls, load_path: str):
        tk = Tokenizer.from_file(load_path)
        return cls(tk)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean",
            )
        return logits, loss


def train_load_tokeniser(
    input_titles: list[str],
    vocab_size: int,
    eos_token: str,
    save_path: str = "./artefacts/bpe_tokeniser.json",
    train=True,
):
    if train:
        print("Train tokenizer")
        tok = BPETokenizer(
            train_tokenizer(input_titles, vocab_size, eos_token=eos_token)
        )
        tok.save(save_path)
    else:
        if os.path.exists(save_path):
            tok = BPETokenizer.load(save_path)
        else:
            raise FileNotFoundError(f"file {save_path} doesn't exist")
    return tok


@dataclass
class RunConfigs:
    tune: bool = False
    train: bool = True
    use_default: bool = True
    train_tok: bool = False


@dataclass
class StorageConfigs:
    url: str = "hyperparameter.db"
    engine: dict = field(
        default_factory=lambda: {"pool_size": 20, "connect_args": {"timeout": 10}}
    )


@dataclass
class TuneConfigs:
    study_name: str = "hptune"
    direction: str = "minimize"
    load_if_exists: bool = True


@dataclass
class Configs:
    run_configs: RunConfigs = field(default_factory=RunConfigs)
    tune_configs: TuneConfigs = field(default_factory=TuneConfigs)
    storage: StorageConfigs = field(default_factory=StorageConfigs)
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def load_configs(file_path: str = "train_config.yaml"):
    if os.path.exists(file_path):
        with open(file_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"no config file use default")
        cfg = {}

    return Configs(
        run_configs=RunConfigs(**cfg.get("run_configs", {})),
        tune_configs=TuneConfigs(**cfg.get("tune_configs", {})),
        storage=StorageConfigs(**cfg.get("storage", {})),
        hyperparameters=cfg.get("hyperparameters", {}),
    )


def suggest_from_config(trial, cfg: dict[str, Any]):
    params = {}
    for name, spec in cfg.items():
        t = spec["type"]
        if t == "loguniform":
            low, high = spec["range"]
            params[name] = trial.suggest_float(name, float(low), float(high), log=True)
        elif t == "uniform":
            low, high = spec["range"]
            params[name] = trial.suggest_float(name, float(low), float(high))
        elif t == "int":
            low, high = spec["range"]
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["values"])
        else:
            raise ValueError(f"Unknown param type: {t}")
    return params


def make_lr_lambda(warmup_steps: int, max_steps: int, min_lr: float=1e-5, peak_lr: float=1.1e-3):
    floor = min_lr / peak_lr

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        
        # Progress through decay
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        # Cosine decay from 1 → floor
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine_decay

    return lr_lambda


def main():

    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global logger
    logger = configure_logging(args.log_file)

    cfg = load_configs()

    if torch.backends.mps.is_available():
        device = "mps"  # Use MPS
        print("MPS is available and will be used!")
    elif torch.cuda.is_available():
        device = "cuda"  # Use GPU
        print("GPU is available and will be used!")
    else:
        device = "cpu"  # Fallback to CPU
        print("MPS is not available. Using CPU.")

    logger.log("device_info", device=device)

    if cfg.run_configs.tune or cfg.run_configs.train:

        train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

        eos_token = "<eos>"
        tok = train_load_tokeniser(
            train_titles + val_titles,
            args.vocab_size,
            eos_token=eos_token,
            train=cfg.run_configs.train_tok,
        )

        # diagnose tokeniser
        # stats = diagnose_tokenizer(tok, train_titles+val_titles)


        # profile the tokeniser
        # stats = profile_tokenizer(tok, train_titles+val_titles)
        # print(stats)

        train_text = eos_token.join(train_titles) + eos_token
        val_text = eos_token.join(val_titles) + eos_token

        train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
        val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    def train_eval(
        lr=args.lr, block_size=args.block_size, dropout=args.dropout, epochs=3
    ):

        batches = len(train_ids) // (block_size * args.batch_size)
        max_steps = epochs * batches
        eval_interval = batches // args.evals_per_epoch
        logger.log(
            "dataset_info",
            titles_count=len(train_titles),
            epochs=epochs,
            batches_per_epoch=batches,
            tokens_per_epoch=len(train_ids),
            vocab_size=tok.vocab_size,
        )

        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_model=args.d_model,
            dropout=dropout,
        )
        model = GPT(cfg).to(device)
        # plot model architect
        # plot_model_highlevel(model, [[args.batch_size, block_size], [args.batch_size, block_size]], [torch.long, torch.long])

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log("model_info", parameters_count=model_params)

        decay, no_decay = [], []
        for n,p in model.named_parameters():
            if not p.requires_grad: 
                continue
            if any(k in n for k in ["bias","norm","Norm","embedding","embeddings"]):
                no_decay.append(p)
            else:
                decay.append(p)

        base_opt = torch.optim.AdamW
        opt = SAM(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            base_opt,
            rho=0.05,
            adaptive=True,
            lr=lr,
            betas=(0.9, 0.95),
        )

        warmup_steps = int(max_steps * args.warmup_ratio)

        lr_lambda = make_lr_lambda(warmup_steps, max_steps, min_lr=args.min_lr, peak_lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.base_optimizer, lr_lambda)

        def evaluate():
            model.eval()
            losses = 0.0
            with torch.no_grad():
                for xb, yb in iter_full_split(
                    val_ids, block_size, args.batch_size, device
                ):
                    logits, _ = model(xb, yb)
                    B, T, V = logits.size()
                    loss = F.cross_entropy(
                        logits.view(-1, V), yb.view(-1), reduction="sum"
                    )
                    losses += loss.item()
            model.train()
            return losses / len(val_text)

        writer = SummaryWriter(log_dir="runs/gpt-2")
        ptr = 0
        step = 0
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{epochs}"):
                step += 1
                xb, yb, ptr = get_batch(
                    train_ids, ptr, block_size, args.batch_size, device
                )

                def closure():
                    _, loss = model(xb, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    return loss


                _, loss = model(xb, yb)
                loss.backward()
                opt.step(closure)
                opt.zero_grad()
                scheduler.step()

                # Log learning rate
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Learning Rate", current_lr, step)


                elapsed = time.time() - t0
                logger.log(
                    "training_step",
                    step=step,
                    max_steps=max_steps,
                    loss=loss.item(),
                    elapsed_time=elapsed,
                    prnt=False,
                )

                if step == 1 or step % eval_interval == 0 or step == max_steps:
                    val_loss = evaluate()
                    logger.log(
                        "validation_step",
                        step=step,
                        max_steps=max_steps,
                        loss=val_loss,
                        elapsed_time=elapsed,
                    )

                    writer.add_scalar("Loss/Train", loss.item(), step)
                    writer.add_scalar("Loss/Validation", val_loss, step)

        return val_loss

    def objective(trial):
        hp_cfg = cfg.hyperparameters
        if hp_cfg == {}:
            raise ValueError(f"Must provide hyperparameters for tial")

        hp = suggest_from_config(trial, hp_cfg)
        loss = train_eval(**hp)

        return loss

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{cfg.storage.url}",
        engine_kwargs=cfg.storage.engine,
    )

    if cfg.run_configs.tune:
        study = optuna.create_study(
            direction=cfg.tune_configs.direction,
            storage=storage,
            study_name=cfg.tune_configs.study_name,
            load_if_exists=cfg.tune_configs.load_if_exists,
        )  # minimise val loss
        study.optimize(objective, n_trials=1)
    elif cfg.run_configs.train:
        hyperparams_dict = vars(args)
        if not cfg.run_configs.use_default:
            if os.path.exists(cfg.storage.url):
                study = optuna.load_study(
                    storage=storage, study_name=cfg.tune_configs.study_name
                )
            else:
                raise FileNotFoundError(f"db file {cfg.storage.url} not found")
            hyperparams_dict.update(study.best_trial.params)
        logger.log("hyperparameters_configured", **hyperparams_dict)

        if cfg.run_configs.use_default:
            train_eval(epochs=args.epochs)
        else:
            train_eval(**study.best_trial.params, epochs=args.epochs)


if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, "file_handler"):
            logger.file_handler.close()
