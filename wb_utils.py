import numpy as np
import plotly.express as px
import torch
from torch import Tensor

import einops
import torch as t
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots


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


## superposition and saes


red = plt.get_cmap("coolwarm")(0.0)
blue = plt.get_cmap("coolwarm")(1.0)
light_grey = np.array([15 / 16, 15 / 16, 15 / 16, 1.0])
red_grey_blue_cmap = LinearSegmentedColormap.from_list(
    "modified_coolwarm",
    np.vstack([np.linspace(red, light_grey, 128), np.linspace(light_grey, blue, 128)]),
)


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


def wb_line(y: Tensor | list, renderer=None, **kwargs):
    """
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    """
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size  # type: ignore
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode="array",
            tickvals=kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext=tickvals,
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    hovertext = kwargs_pre.pop("hovertext", None)
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = (
            list(map(to_numpy, y))
            if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float))
            else to_numpy(y)
        )  # type: ignore
        names = kwargs_pre.pop("names", None)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
    if hovertext is not None:
        ht = fig.data[0].hovertemplate
        fig.for_each_trace(
            lambda trace: trace.update(hovertext=hovertext, hovertemplate="%{hovertext}<br>" + ht)
        )

    return fig if return_fig else fig.show(renderer=renderer)


def sort_W_by_monosemanticity(
    W: Float[Tensor, "feats d_hidden"],
) -> tuple[Float[Tensor, "feats d_hidden"], int]:
    """
    Rearranges the columns of the tensor (i.e. rearranges neurons) in descending order of
    their monosemanticity (where we define monosemanticity as the largest fraction of this
    neuron's norm which is a single feature).

    Also returns the number of "monosemantic features", which we (somewhat arbitrarily)
    define as the fraction being >90% of the total norm.
    """
    norm_by_neuron = W.pow(2).sum(dim=0)
    monosemanticity = W.abs().max(dim=0).values / (norm_by_neuron + 1e-6).sqrt()

    column_order = monosemanticity.argsort(descending=True).tolist()

    n_monosemantic_features = int((monosemanticity.abs() > 0.99).sum().item())

    return W[:, column_order], n_monosemantic_features


def rearrange_full_tensor(
    W: Float[Tensor, "inst d_hidden feats"],
):
    """
    Same as above, but works on W in its original form, and returns a list of
    number of monosemantic features per instance.
    """
    n_monosemantic_features_list = []

    for i, W_inst in enumerate(W):
        W_inst_rearranged, n_monosemantic_features = sort_W_by_monosemanticity(W_inst.T)
        W[i] = W_inst_rearranged.T
        n_monosemantic_features_list.append(n_monosemantic_features)

    return W, n_monosemantic_features_list


def get_viridis_str(v: float) -> str:
    r, g, b, a = plt.get_cmap("viridis")(v)
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"rgb({r}, {g}, {b})"


def clamp(x: float, min_val: float, max_val: float) -> float:
    return min(max(x, min_val), max_val)


def plot_features_in_Nd(
    W: Float[Tensor, "inst d_hidden feats"],
    height: int,
    width: int,
    title: str | None = None,
    subplot_titles: list[str] | None = None,
    neuron_plot: bool = False,
):
    n_instances, d_hidden, n_feats = W.shape

    W = W.detach().cpu()

    # Rearrange to align with standard basis
    W, n_monosemantic_features = rearrange_full_tensor(W)

    # Normalize W, i.e. W_normed[inst, i] is normalized i-th feature vector
    W_normed = W / (1e-6 + t.linalg.norm(W, 2, dim=1, keepdim=True))

    # We get interference[i, j] = sum_{j!=i} (W_normed[i] @ W[j]) (ignoring the instance dimension)
    # because then we can calculate superposition by squaring & summing this over j
    interference = einops.einsum(
        W_normed,
        W,
        "instances hidden feats_i, instances hidden feats_j -> instances feats_i feats_j",
    )
    interference[:, range(n_feats), range(n_feats)] = 0

    # Now take the sum, and sqrt (we could just as well not sqrt)
    # Heuristic: polysemanticity is zero if it's orthogonal to all else, one if it's perfectly aligned with any other single vector
    polysemanticity = einops.reduce(
        interference.pow(2),
        "instances feats_i feats_j -> instances feats_i",
        "sum",
    ).sqrt()
    colors = [
        [get_viridis_str(v.item()) for v in polysemanticity_for_this_instance]
        for polysemanticity_for_this_instance in polysemanticity
    ]

    # Get the norms (this is the bar height)
    W_norms = einops.reduce(
        W.pow(2),
        "instances hidden feats -> instances feats",
        "sum",
    ).sqrt()

    # We need W.T @ W for the heatmap (unless this is a neuron plot, then we just use w)
    if not (neuron_plot):
        WtW = einops.einsum(
            W,
            W,
            "instances hidden feats_i, instances hidden feats_j -> instances feats_i feats_j",
        )
        imshow_data = WtW.numpy()
    else:
        imshow_data = einops.rearrange(W, "instances hidden feats -> instances feats hidden").numpy()

    # Get titles (if they exist). Make sure titles only apply to the bar chart in each row
    titles = ["Heatmap of " + ("W" if neuron_plot else "W<sup>T</sup>W")] * n_instances + [
        "Neuron weights<br>stacked bar plot" if neuron_plot else "Feature norms"
    ] * n_instances  # , ||W<sub>i</sub>||
    if subplot_titles is not None:
        for i, st in enumerate(subplot_titles):
            titles[i] = st + "<br>" + titles[i]

    total_height = 0.9 if title is None else 0.8
    if neuron_plot:
        heatmap_height_fraction = clamp(n_feats / (d_hidden + n_feats), 0.5, 0.75)
    else:
        heatmap_height_fraction = 1 - clamp(n_feats / (30 + n_feats), 0.5, 0.75)
    row_heights = [
        total_height * heatmap_height_fraction,
        total_height * (1 - heatmap_height_fraction),
    ]

    n_rows = 2
    n_cols = n_instances

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.1 if neuron_plot else 0.05,
        row_heights=row_heights,
        subplot_titles=titles,
    )
    for inst in range(n_instances):
        # (1) Add bar charts
        # If it's the non-neuron plot then x = features, y = norms of those features. If it's the
        # neuron plot, our x = neurons (d_hidden), y = the loadings of features on those neurons. In
        # both cases, colors = polysemanticity of features, which we've already computed
        if neuron_plot:
            for feat in range(n_feats):
                fig.add_trace(
                    go.Bar(
                        x=t.arange(d_hidden),
                        y=W[inst, :, feat],
                        marker=dict(color=[colors[inst][feat]] * d_hidden),
                        width=0.9,
                    ),
                    col=1 + inst,
                    row=2,
                )
        else:
            fig.add_trace(
                go.Bar(
                    y=t.arange(n_feats).flip(0),
                    x=W_norms[inst],
                    marker=dict(color=colors[inst]),
                    width=0.9,
                    orientation="h",
                ),
                col=1 + inst,
                row=2,
            )
        # (2) Add heatmap
        # Code is same for neuron plot vs no neuron plot, although data is different: W.T @ W vs W
        fig.add_trace(
            go.Image(
                z=red_grey_blue_cmap((1 + imshow_data[inst]) / 2, bytes=True),
                colormodel="rgba256",
                customdata=imshow_data[inst],
                hovertemplate="""In: %{x}<br>\nOut: %{y}<br>\nWeight: %{customdata:0.2f}""",
            ),
            col=1 + inst,
            row=1,
        )

    if neuron_plot:
        # Stacked plots to allow for all features to be seen
        fig.update_layout(barmode="relative")

        # Weird naming convention for subplots, make sure we have a list of the subplot names for bar charts so we can iterate through them
        n0 = 1 + n_instances
        fig_indices = [str(i) if i != 1 else "" for i in range(n0, n0 + n_instances)]

        for inst in range(n_instances):
            fig["layout"][f"yaxis{fig_indices[inst]}_range"] = [-6, 6]  # type: ignore

            # Add the background colors
            row, col = (2, 1 + inst)
            fig.add_vrect(
                x0=-0.5,
                x1=-0.5 + n_monosemantic_features[inst],
                fillcolor="#440154",
                line_width=0.0,
                opacity=0.2,
                col=col,  # type: ignore
                row=row,  # type: ignore
                layer="below",
            )
            fig.add_vrect(
                x0=-0.5 + n_monosemantic_features[inst],
                x1=-0.5 + d_hidden,
                fillcolor="#fde725",
                line_width=0.0,
                opacity=0.2,
                col=col,  # type: ignore
                row=row,  # type: ignore
                layer="below",
            )

    else:
        # Add annotation of "features" on the y-axis of the bar plot
        fig_indices = [str(i) if i != 1 else "" for i in range(n_instances + 1, 2 * n_instances + 1)]
        for inst in range(n_instances):
            fig.add_annotation(
                text="Features ➔",  # ➤→⮕🡒➜
                xref=f"x{fig_indices[inst]} domain",
                yref=f"y{fig_indices[inst]} domain",
                x=-0.13,
                y=0.99,  # Positioning the annotation outside the first bar plot subfigure
                showarrow=False,
                font=dict(size=12),
                textangle=90,  # Set the text angle to 90 degrees for vertical text
            )

    # Add a horizontal line at the point where n_features = d_hidden (in non-neuron plot). After this point,
    # we must have superposition if we represent all features.
    for annotation in fig.layout.annotations:
        annotation.font.size = 13
    if not neuron_plot:
        fig.add_hline(
            y=n_feats - d_hidden - 0.5,
            line=dict(width=0.5),
            opacity=1.0,
            row=2,  # type: ignore
            annotation_text=f" d_hidden={d_hidden}",
            annotation_position="bottom left",  # "bottom"
            annotation_font_size=11,
        )

    # fig.update_traces(marker_size=1)
    fig.update_layout(
        showlegend=False,
        width=width,
        height=height,
        margin=dict(t=40 if title is None else 110, b=40, l=50, r=40),
        plot_bgcolor="#eee",
        title=title,
        title_y=0.95,
        # template="simple_white",
    )

    fig.update_xaxes(showticklabels=False, showgrid=False)  # visible=False
    fig.update_yaxes(showticklabels=False, showgrid=False)

    fig.show()
