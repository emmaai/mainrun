import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from train import train_load_tokeniser, iter_full_split, GPTConfig, GPT

seed = 1337
num_titles = 100_000

def load_titles():
    ds = load_dataset(
        "julien040/hacker-news-posts", split="train", cache_dir="./data"
    ).shuffle(seed=seed)
    ds = ds.skip(num_titles)
    titles = [row["title"].strip() for row in ds.take(2*num_titles)]
    print(titles[:5])
    return titles 


def evaluate_model(model, tok, texts, block_size=64, device="mps"):

    eos_token = "<eos>"

    val_text = eos_token.join(texts) + eos_token
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    losses = 0

    model.eval()
    
    with torch.no_grad():
        for xb, yb in iter_full_split(
                val_ids, block_size, 64, device
            ):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(
                    logits.view(-1, V), yb.view(-1), reduction="sum"
                )
                losses += loss.item()
        model.train()
    avg_loss = losses / len(val_text)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

if __name__ == "__main__":
    tok = train_load_tokeniser(None, None, None, save_path="artefacts/bpe_tokeniser.json", train=False)
    val_text = load_titles()

    results = {}
    for model_file in ["models/model_midsize.pth", "models/model_smallsize.pth"]:
        ckpt = torch.load(model_file, map_location="mps")
        cfg = ckpt["config"]
        block_size = cfg["block_size"]
        cfg = GPTConfig(
            vocab_size=cfg["vocab_size"],
            block_size=block_size,
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            d_model=cfg["d_model"],
            dropout=cfg["dropout"],
        )
        model = GPT(cfg).to("mps")
        model.load_state_dict(ckpt["model_state"])

        avg_loss, ppl = evaluate_model(model, tok, val_text, block_size=block_size)
        results[model_file] = {"loss": avg_loss, "ppl": ppl} 
        print(results)
