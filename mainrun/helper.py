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
