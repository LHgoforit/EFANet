import os
import torch
from contextlib import contextmanager


@contextmanager
def torch_profiler(output_dir="profiling", use_kineto=True, record_shapes=True, with_stack=True, export_chrome=True):
    """
    Context manager for PyTorch profiling.

    Args:
        output_dir (str): Directory to save profiling results.
        use_kineto (bool): Use newer Kineto backend (recommended for PyTorch >=1.8).
        record_shapes (bool): Track tensor shapes.
        with_stack (bool): Capture stack trace for each op.
        export_chrome (bool): Export timeline in Chrome Trace format.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(output_dir),
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_flops=True
        ) as prof:
            yield prof

    except ImportError:
        # Fallback for older PyTorch
        with torch.autograd.profiler.profile(
            use_cuda=torch.cuda.is_available(),
            record_shapes=record_shapes
        ) as prof:
            yield prof


def print_profiler_summary(prof, topk=20, sort_by="cuda_time_total"):
    """
    Print a sorted summary table of profiler results.

    Args:
        prof: The profiler object returned by context.
        topk (int): Number of top operations to print.
        sort_by (str): Metric to sort by (e.g., 'cuda_time_total', 'cpu_time_total').
    """
    try:
        print(prof.key_averages().table(sort_by=sort_by, row_limit=topk))
    except Exception as e:
        print(f"[Profiler] Failed to print summary: {e}")


def export_chrome_trace(prof, filename="trace.json"):
    """
    Export trace to Chrome Trace Viewer format.

    Args:
        prof: Profiler object.
        filename (str): Output file path.
    """
    try:
        prof.export_chrome_trace(filename)
        print(f"[Profiler] Chrome trace saved to: {filename}")
    except Exception as e:
        print(f"[Profiler] Failed to export Chrome trace: {e}")
