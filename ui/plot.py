import numpy as np
import plotly.graph_objects as go

from color import colors20


def cp_factor_scatter(
    cp_factors,
    labels,
    mode=0,
    x_factor_index=0,
    y_factor_index=1,
    colors=colors20,
    label_names=None,
):
    fig = go.Figure()
    for label in np.unique(labels):
        fig.add_trace(
            go.Scatter(
                x=cp_factors[mode][:, x_factor_index][labels == label],
                y=cp_factors[mode][:, y_factor_index][labels == label],
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors[label],
                    line=dict(width=0.5, color="#aaaaaa"),
                ),
                name=(
                    label_names[label]
                    if label_names is not None
                    else f"Group {label+1}  "  # NOTE: adding space for nicer placing
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        autosize=True,
        plot_bgcolor="#ffffff",
        xaxis=dict(
            showticklabels=False, showgrid=False, showline=False, zeroline=False
        ),
        yaxis=dict(
            showticklabels=False, showgrid=False, showline=False, zeroline=False
        ),
        coloraxis=dict(colorbar=dict(title="Color Index")),
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(
            font=dict(size=10),
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        dragmode="lasso",
    )
    return fig


def cp_factor_bar(
    cp_factors,
    mode,
    factor_index,
    xlabel="Component",
    ylabel="Value",
    show_x_ticks=True,
):
    values = cp_factors[mode][:, factor_index]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=np.arange(len(values)) + 1, y=values, marker=dict(color="gray"))
    )
    fig.update_layout(
        autosize=True,
        plot_bgcolor="#ffffff",
        showlegend=False,
        margin=dict(t=0, b=2, l=0, r=0),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            gridcolor="#444444",
            linecolor="#444444",
            zerolinecolor="#444444",
            tickfont=dict(size=10),
            showticklabels=True if show_x_ticks else False,
            title=(None if show_x_ticks else dict(text=xlabel, font=dict(size=10))),
        ),
        yaxis=dict(
            range=[-np.max(np.abs(values)), np.max(np.abs(values))],
            showticklabels=True,
            showgrid=False,
            showline=True,
            zeroline=True,
            gridcolor="#888888",
            linecolor="#888888",
            zerolinecolor="#888888",
            zerolinewidth=1,
            tickfont=dict(size=10),
            title=dict(text=ylabel, font=dict(size=10)),
        ),
    )

    return fig


def heatmap(
    U,
    x_ticks=None,
    y_ticks=None,
    xlabel=None,
    ylabel=None,
    colorbar_legend=None,
    colorscale="RdBu_r",
    filter_out_range=None,
):
    max_abs_value = np.abs(U).max()

    remaining_indices = np.ones(U.shape[0], dtype=bool)
    if filter_out_range is not None:
        # < 0.05 consider the user actually don't want to filter
        if filter_out_range[1] - filter_out_range[0] >= 0.05:
            remaining_indices = np.any(
                (U <= filter_out_range[0]) + (U >= filter_out_range[1]), axis=1
            )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=U[remaining_indices],
            x=x_ticks if x_ticks is not None else np.arange(U.shape[1]) + 1,
            y=(
                np.array(y_ticks)[remaining_indices]
                if y_ticks is not None
                else np.array(np.arange(U.shape[0]) + 1, dtype=str)[remaining_indices]
            ),
            colorscale=colorscale,
            zmin=-max_abs_value,
            zmax=max_abs_value,
            colorbar=dict(
                title=colorbar_legend,
                thickness=20,
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=10),
                x=1.1,
            ),
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        autosize=True,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
        xaxis=dict(
            tick0=1,
            dtick=1,
            tickfont=dict(size=10),
            title=dict(text=xlabel, font=dict(size=10)),
        ),
        yaxis=dict(tickfont=dict(size=10), title=dict(text=ylabel, font=dict(size=10))),
    )

    return fig
