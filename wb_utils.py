import numpy as np
import plotly.express as px
import torch
from torch import Tensor


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def plot_loss_difference(log_probs, rep_str, seq_len, filename: str | None = None):
    fig = px.line(
        to_numpy(log_probs),
        hover_name=rep_str[1:],
        title=f"Per token log prob on correct token, for sequence of length {seq_len}*2 (repeated twice)",
        labels={"index": "Sequence position", "value": "Log prob"},
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=seq_len - 0.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len - 0.5, x1=2 * seq_len - 1, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()
    if filename is not None:
        fig.write_html(filename)
