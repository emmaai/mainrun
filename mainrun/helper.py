from torchview import draw_graph
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def plot_model_highlevel(
    model,
    input_size,
    dtypes,
    save_path="model_highlevel",
    expand_nested=True,
    roll=True,
    model_name="Model",
    format="png"
):
    """
    Plot a high-level architecture diagram using torchview.

    Parameters:
    - model: nn.Module
    - input_size: tuple or list of tuples
        Shape(s) of input(s), e.g., (batch, seq_len, dim) or [(seq_len, batch, dim), (tgt_len, batch, dim)]
    - dytpes: list of torch.type 
        set the types of input tensor.
    - save_path: str
        File path to save the diagram (without extension).
    - expand_nested: bool
        Whether to expand nested nn.Modules (e.g., TransformerEncoder).
    - roll: bool
        Use horizontal layout.
    - model_name: str
        Name displayed on the graph.
    - format: str
        File format: 'png', 'pdf', etc.
    """
    model.eval()

    graph = draw_graph(
        model,
        input_size=input_size,
        dtypes=dtypes,
        expand_nested=expand_nested,
        roll=roll,
        graph_name=model_name
    )

    graph.visual_graph.render(save_path, format=format, cleanup=True)
    print(f"High-level architecture saved to: {save_path}.{format}")

def profile_tokenizer(tokenizer, texts: List[str], max_length: int = 1024, top_k: int = 20) -> Dict[str, Any]:
    """
    Profile a HuggingFace tokenizer on a dataset.

    Args:
        tokenizer: BPETokerniser.
        texts (List[str]): Dataset of raw text samples.
        max_length (int): Model context length (default 1024 for GPT-2).
        top_k (int): How many most common tokens to display.

    Returns:
        Dict with profiling statistics.
    """
    lengths = []
    token_counter = Counter()

    for txt in texts:
        tokens = tokenizer.encode(txt)
        lengths.append(len(tokens))
        token_counter.update(tokens)

    lengths = np.array(lengths)
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "num_samples": len(texts),
        "avg_tokens": float(np.mean(lengths)),
        "median_tokens": float(np.median(lengths)),
        "max_tokens": int(np.max(lengths)),
        "min_tokens": int(np.min(lengths)),
        "pct_over_context": float(np.mean(lengths > max_length) * 100),
    }

    # Plot length histogram
    plt.hist(lengths, bins=50)
    plt.axvline(max_length, color='red', linestyle='--', label=f"context={max_length}")
    plt.xlabel("Sequence length (# tokens)")
    plt.ylabel("Count")
    plt.title("Token length distribution")
    plt.legend()
    plt.show()

    return stats

def diagnose_tokenizer(tokenizer, texts, sample_size=5000, max_plot_tokens=100):
    """
    Diagnose a tokenizer on a given text corpus.
    
    Args:
        tokenizer: HuggingFace tokenizer (or any object with `encode` method returning list[int])
        texts: list of strings
        sample_size: number of texts to sample for diagnostics
        max_plot_tokens: how many tokens to show in frequency plot
    
    Returns:
        stats dict
    """
    # sample corpus
    if sample_size < len(texts):
        import random
        texts = random.sample(texts, sample_size)
    
    all_ids = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    unk_count = 0
    
    lengths = []
    
    for t in texts:
        ids = tokenizer.encode(t)
        all_ids.extend(ids)
        lengths.append(len(ids))
        if unk_token_id is not None:
            unk_count += sum(1 for i in ids if i == unk_token_id)
    
    # frequency analysis
    freq = Counter(all_ids)
    vocab_size = tokenizer.vocab_size
    total_tokens = len(all_ids)
    
    # UNK rate
    unk_rate = unk_count / total_tokens if total_tokens > 0 else 0.0
    
    # sequence length stats
    lengths = np.array(lengths)
    
    # Zipf curve plot
    sorted_freq = [c for _, c in freq.most_common(max_plot_tokens)]
    plt.figure(figsize=(8,4))
    plt.plot(sorted_freq)
    plt.title("Top token frequencies")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()
    
    stats = {
        "vocab_size": vocab_size,
        "total_tokens_encoded": total_tokens,
        "unk_rate": unk_rate,
        "avg_seq_len": float(np.mean(lengths)),
        "median_seq_len": float(np.median(lengths)),
        "p95_seq_len": float(np.percentile(lengths, 95)),
        "num_unique_tokens_seen": len(freq),
        "coverage_ratio": len(freq) / vocab_size
    }
    
    return stats

