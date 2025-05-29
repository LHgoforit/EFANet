import os
import json
import numpy as np

def save_metrics_to_json(metrics_dict, save_path):
    """
    Save evaluation metrics dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[Metrics] Saved metrics to {save_path}")

def load_metrics_from_json(path):
    """
    Load evaluation metrics dictionary from a JSON file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, 'r') as f:
        metrics = json.load(f)
    return metrics

def generate_latex_table(metrics_dict, caption="Quantitative Results", label="tab:metrics"):
    """
    Convert a metrics dictionary into a LaTeX-formatted table string.
    Example input:
    {
        "EFANet": {"PSNR": 27.54, "SSIM": 0.843, "LPIPS": 0.192},
        "GFPGAN": {"PSNR": 25.31, "SSIM": 0.812, "LPIPS": 0.241}
    }
    """
    methods = list(metrics_dict.keys())
    metrics = list(next(iter(metrics_dict.values())).keys())

    table = []
    table.append("\\begin{table}[ht]\n\\centering")
    table.append("\\caption{" + caption + "}")
    table.append("\\label{" + label + "}")
    col_spec = "l" + "c" * len(metrics)
    table.append("\\begin{tabular}{" + col_spec + "}")
    table.append("\\toprule")
    header = ["Method"] + metrics
    table.append(" & ".join(header) + " \\\\")
    table.append("\\midrule")

    for method in methods:
        row = [method] + [f"{metrics_dict[method][m]:.3f}" for m in metrics]
        table.append(" & ".join(row) + " \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")

    return "\n".join(table)

def summarize_metrics(metrics_list, method_names):
    """
    Summarize a list of metric dictionaries (per image) into mean values per method.
    Example input:
    [
        {"EFANet": {"PSNR": 27.2, "SSIM": 0.83, "LPIPS": 0.19}, ...},
        {"EFANet": {"PSNR": 27.9, "SSIM": 0.85, "LPIPS": 0.18}, ...}
    ]
    """
    summary = {method: {} for method in method_names}
    for method in method_names:
        all_values = {}
        for entry in metrics_list:
            for k, v in entry[method].items():
                all_values.setdefault(k, []).append(v)
        for k, v in all_values.items():
            summary[method][k] = float(np.mean(v))
    return summary
