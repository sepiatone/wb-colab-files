import numpy as np
import plotly.express as px
import torch
from torch import Tensor
from jaxtyping import Float
from matplotlib import pyplot as plt


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


def convert_tokens_to_string(model, tokens, batch_index=0):
    """
    Helper function to convert tokens into a list of strings, for printing.
    """
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]


def get_viridis(v: float) -> tuple[float, float, float]:
    r, g, b, a = plt.get_cmap("viridis")(v)
    return (r, g, b)


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

update_layout_set = {
    "xaxis_range",
    "yaxis_range",
    "hovermode",
    "xaxis_title",
    "yaxis_title",
    "colorbar",
    "colorscale",
    "coloraxis",
    "title_x",
    "bargap",
    "bargroupgap",
    "xaxis_tickformat",
    "yaxis_tickformat",
    "title_y",
    "legend_title_text",
    "xaxis_showgrid",
    "xaxis_gridwidth",
    "xaxis_gridcolor",
    "yaxis_showgrid",
    "yaxis_gridwidth",
    "yaxis_gridcolor",
    "showlegend",
    "xaxis_tickmode",
    "yaxis_tickmode",
    "margin",
    "xaxis_visible",
    "yaxis_visible",
    "bargap",
    "bargroupgap",
    "coloraxis_showscale",
    "xaxis_tickangle",
    "yaxis_scaleanchor",
    "xaxis_tickfont",
    "yaxis_tickfont",
}


def wb_imshow(tensor: torch.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size  # type: ignore
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label  # type: ignore
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(text=_text, texttemplate="%{text}", textfont={"size": 12})
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1 + n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle  # type: ignore
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})


def plot_logit_attribution(model, logit_attr: torch.Tensor, tokens: torch.Tensor, title: str = "", filename: str | None = None):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    fig = wb_imshow(
        to_numpy(logit_attr),  # type: ignore
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        title=title if title else None,
        height=100 + (30 if title else 0) + 15 * len(y_labels),
        width=24 * len(x_labels),
        return_fig=True,
    )
    fig.show()
    if filename is not None:
        fig.write_html(filename)


def cast_element_to_nested_list(elem, shape: tuple):
    """
    Creates a nested list of shape `shape`, where every element is `elem`.
    Example: ("a", (2, 2)) -> [["a", "a"], ["a", "a"]]
    """
    if len(shape) == 0:
        return elem
    return [cast_element_to_nested_list(elem, shape[1:])] * shape[0]


def plot_features_in_2d(
    W: Float[Tensor, "*inst d_hidden feats"] | list[Float[Tensor, "d_hidden feats"]],
    colors: Float[Tensor, "inst feats"] | list[str] | list[list[str]] | None = None,
    title: str | None = None,
    subplot_titles: list[str] | None = None,
    allow_different_limits_across_subplots: bool = False,
    n_rows: int | None = None,
):
    """
    Visualises superposition in 2D.

    If values is 4D, the first dimension is assumed to be timesteps, and an animation is created.
    """
    # Convert W into a list of 2D tensors, each of shape [feats, d_hidden=2]
    if isinstance(W, Tensor):
        if W.ndim == 2:
            W = W.unsqueeze(0)
        n_instances, d_hidden, n_feats = W.shape
        n_feats_list = []
        W = W.detach().cpu()
    else:
        # Hacky case which helps us deal with double descent exercises (this is never used outside of those exercises)
        assert all(w.ndim == 2 for w in W)
        n_feats_list = [w.shape[1] for w in W]
        n_feats = max(n_feats_list)
        n_instances = len(W)
        W = [w.detach().cpu() for w in W]

    W_list: list[Tensor] = [W_instance.T for W_instance in W]

    # Get some plot characteristics
    limits_per_instance = (
        [w.abs().max() * 1.1 for w in W_list]
        if allow_different_limits_across_subplots
        else [1.5 for _ in range(n_instances)]
    )
    linewidth, markersize = (1, 4) if (n_feats >= 25) else (1.5, 6)

    # Maybe break onto multiple rows
    if n_rows is None:
        n_rows, n_cols = 1, n_instances
        row_col_tuples = [(0, i) for i in range(n_instances)]
    else:
        n_cols = n_instances // n_rows
        row_col_tuples = [(i // n_cols, i % n_cols) for i in range(n_instances)]

    # Convert colors into a 2D list of strings, with shape [instances, feats]
    if colors is None:
        colors_list = cast_element_to_nested_list("black", (n_instances, n_feats))
    elif isinstance(colors, str):
        colors_list = cast_element_to_nested_list(colors, (n_instances, n_feats))
    elif isinstance(colors, list):
        # List of strings -> same for each instance and feature
        if isinstance(colors[0], str):
            assert len(colors) == n_feats
            colors_list = [colors for _ in range(n_instances)]
        # List of lists of strings -> different across instances & features (we broadcast)
        else:
            colors_list = []
            for i, colors_for_instance in enumerate(colors):
                assert len(colors_for_instance) in (1, n_feats_list[i])
                colors_list.append(colors_for_instance * (n_feats_list[i] if len(colors_for_instance) == 1 else 1))
    elif isinstance(colors, Tensor):
        assert colors.shape == (n_instances, n_feats)
        colors_list = [[get_viridis(v) for v in color] for color in colors.tolist()]

    # Create a figure and axes, and make sure axs is a 2D array
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axs = np.broadcast_to(axs, (n_rows, n_cols))

    # If there are titles, add more spacing for them
    fig.subplots_adjust(bottom=0.2, top=(0.8 if title else 0.9), left=0.1, right=0.9, hspace=0.5)

    # Initialize lines and markers
    for instance_idx, ((row, col), limits_per_instance) in enumerate(zip(row_col_tuples, limits_per_instance)):
        # Get the right axis, and set the limits
        ax = axs[row, col]
        ax.set_xlim(-limits_per_instance, limits_per_instance)
        ax.set_ylim(-limits_per_instance, limits_per_instance)
        ax.set_aspect("equal", adjustable="box")

        # Add all the features for this instance
        _n_feats = n_feats if len(n_feats_list) == 0 else n_feats_list[instance_idx]
        for feature_idx in range(_n_feats):
            x, y = W_list[instance_idx][feature_idx].tolist()
            color = colors_list[instance_idx][feature_idx]
            ax.plot([0, x], [0, y], color=color, lw=linewidth)[0]
            ax.plot([x, x], [y, y], color=color, marker="o", markersize=markersize)[0]

        # Add titles & subtitles
        if title:
            fig.suptitle(title, fontsize=15)
        if subplot_titles:
            axs[row, col].set_title(subplot_titles[instance_idx], fontsize=12)

    plt.show()
