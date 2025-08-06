import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import numpy as np
import tensorly as tl
from scipy.cluster.hierarchy import linkage, leaves_list
from tulca import TULCA

from color import colors10, colors20
from plot import cp_factor_scatter, cp_factor_bar, heatmap

states = {"selected_indices": []}


def reorder_matrix_by_similarity(
    U, method="average", metric="cosine", apply_to_rows=False
):
    col_linkage_matrix = linkage(
        U.T, method=method, metric=metric, optimal_ordering=True
    )
    col_order = leaves_list(col_linkage_matrix)
    row_order = np.arange(U.shape[0])
    if apply_to_rows:
        row_linkage_matrix = linkage(
            U, method=method, metric=metric, optimal_ordering=True
        )
        row_order = leaves_list(row_linkage_matrix)

    return U[row_order, :][:, col_order], col_order, row_order


def slider(
    slider_id,
    title,
    color,
    min_val=0,
    max_val=1,
    value=1,
    width=170,
    height=20,
    handle_width=5,
    handle_height=16,
):
    # Plotly dcc only supports a part of styles and needs to use css
    # NOTE: css needs to use "-" based description (background-color) unlike Plotly's captilized one (e.g., backgroundColor)
    slider_css = f"#{slider_id} {{ height: 0px; width: {width}px; }}"
    slider_css += f"#{slider_id} .rc-slider {{ padding-bottom: {height*0.7}px; }}"
    slider_css += f"#{slider_id} .rc-slider-track {{ background: {color}; height: {height}px; border-radius: 0px}}"
    slider_css += f"#{slider_id} .rc-slider-handle {{ background-color: #FFFFFF; border-width: 1px; border-color: #888888; width: {handle_width}px; height: {handle_height}px; border-radius: 1px; box-shadow: none; margin-top: {(height - handle_height)/2}px; }}"
    slider_css += f"#{slider_id} .rc-slider-rail {{ background-color: #FFFFFF }}"

    slider_div = html.Div(
        [
            dcc.Checklist(
                id=slider_id + "_checkbox",
                options=[title],
                style={
                    "display": "table-cell",
                    "fontSize": 10,
                    "width": 80,
                    "verticalAlign": "middle",
                    "accentColor": "#D3D3D3",
                },
                inputStyle={"width": 9, "height": 9, "marginRight": 1},
            ),
            dcc.Slider(
                id=slider_id,
                min=min_val,
                max=max_val,
                value=value,
                marks=None,
                tooltip=None,
            ),
        ],
        style={
            "display": "table",
            "marginLeft": 0,
            "marginBottom": 0,
            "marginTop": 0,
            "height": height * 1.5,
        },
    )

    return slider_div, slider_css


def init_weight_sliders(w_tg, w_bg, w_bw, colors, label_names):
    n_classes = len(w_tg)

    # weight_types = ["w_tg", "w_bg", "w_bw", "misc"]
    # weight_descs = [
    #     "Target weight",
    #     "Background weight",
    #     "Between-class weight",
    #     "Other parameters",
    # ]
    weight_types = ["w_tg", "w_bg", "w_bw"]
    weight_descs = ["Target weight", "Background weight", "Between-class weight"]

    slider_css_path = os.path.join(os.path.dirname(__file__), "assets/slider.css")

    slider_ids = []
    slider_divs = []
    slider_csss = ""
    slider_height = 20 if n_classes <= 5 else 5
    for weight_type, weight_desc in zip(weight_types, weight_descs):
        html_elements = [
            html.P(
                weight_desc,
                id=f"title-{weight_type}",
                style={"marginBottom": 0, "marginTop": 5},
            )
        ]
        if weight_type != "misc":
            if weight_type == "w_tg":
                w = w_tg
            elif weight_type == "w_bg":
                w = w_bg
            else:
                w = w_bw
            for class_id in range(n_classes):
                slider_id = f"slider-{weight_type}-{class_id}"
                slider_title = (
                    label_names[class_id]
                    if label_names is not None
                    else f"Group {class_id + 1}"
                )
                slider_color = (
                    colors[class_id] if class_id < len(colors) else colors[-1]
                )
                slider_div, slider_css = slider(
                    slider_id,
                    slider_title,
                    slider_color,
                    value=w[class_id],
                    height=slider_height,
                    handle_width=5 if n_classes <= 5 else 3,
                    handle_height=(
                        slider_height - 4 if n_classes <= 5 else slider_height
                    ),
                )
                slider_ids.append(slider_id)
                html_elements.append(slider_div)
                slider_csss += slider_css
        # else:
        #     # TODO: make this customizable based on a list of params
        #     slider_id = f"slider-misc-tradeoff"
        #     slider_title = "Trade-off"
        #     slider_color = "#444444"
        #     slider_div, slider_css = slider(
        #         slider_id,
        #         slider_title,
        #         slider_color,
        #         height=slider_height,
        #         handle_width=5 if n_classes <= 5 else 3,
        #         handle_height=(slider_height - 4 if n_classes <= 5 else slider_height),
        #     )
        #     slider_ids.append(slider_id)
        #     html_elements.append(slider_div)
        #     slider_csss += slider_css

        slider_divs.append(html.Div(html_elements))
    with open(slider_css_path, "w") as f:
        f.write(slider_csss)

    return slider_ids, slider_divs


def init_mode_radios(ys, target_mode=0, height=25):
    title = "Comparing mode"
    labels = ["Time", "Instance", "Variable"]
    label_style = {"display": "table-cell", "verticalAlign": "middle"}
    disables = [y is None for y in ys]

    radio_divs = [html.Label(title)]
    options = [
        {"label": html.Div([label], style=label_style), "value": i, "disabled": disable}
        for i, (label, disable) in enumerate(zip(labels, disables))
    ]
    radio_divs.append(
        dcc.RadioItems(
            id="mode-radios",
            options=options,
            value=target_mode,
            inputStyle={
                "display": "table-cell",
                "verticalAlign": "middle",
                "cursor": "pointer",
                "marginRight": 10,
                "height": "100%",
            },
            labelStyle={"display": "table", "height": height},
        )
    )
    return radio_divs


def init_n_components_drowdowns(n_components_list, max_n_components_list, height=25):
    title = "# of components"
    dropdown_ids = [
        "ncomps-time-dropdown",
        "ncomps-inst-dropdown",
        "ncomps-var-dropdown",
    ]
    dropdown_css_path = os.path.join(os.path.dirname(__file__), "assets/dropdown.css")

    dropdown_css = f".Select-value {{ transform: translateY(-8px); height: {height + 5}px; line-height: {height -5}px; }} "
    dropdown_css += ".Select-arrow { transform: translateY(2px); } "
    dropdown_css += (
        f".Select-value-label {{ height: {height + 5}px; line-height: {height-5}px; }} "
    )
    dropdown_css += ".Select-control { width: 100% !important; border-radius: 5px; display: flex; } "
    dropdown_css += ".Select-control .Select-multi-value-wrapper { flex-grow: 2; } "
    with open(dropdown_css_path, "w") as f:
        f.write(dropdown_css)

    dropdown_divs = [html.Label(title)]
    for dropdown_id, value, max_value in zip(
        dropdown_ids,
        n_components_list,
        max_n_components_list,
    ):
        dropdown_divs.append(
            dcc.Dropdown(
                id=dropdown_id,
                options=[str(i + 1) for i in range(max_value)],
                value=str(value),
                clearable=False,
                style={
                    "width": "50%",
                    "margin": "0px auto 5px 0px",
                    "height": height - 5,
                },
            )
        )
    return dropdown_ids, dropdown_divs


def init_scatterplot_axes_dropdowns(n_rankone_tensors, height=25):
    title = "Scatterplot axes"
    dropdown_ids = ["xaxis-factor-index-dropdown", "yaxis-factor-index-dropdown"]
    dropdopn_labels = ["x-axis rank-1 tensor index", "y-axis rank-1 tensor index"]
    values = [1, 2]
    options = np.arange(n_rankone_tensors) + 1

    dropdown_divs = [html.Label(title)]
    for dropdown_id, label, value in zip(dropdown_ids, dropdopn_labels, values):
        dropdown_divs.append(
            html.Div(
                [
                    html.P(
                        label,
                        style={
                            "display": "table-cell",
                            "fontSize": 10,
                            "width": 150,
                            "verticalAlign": "middle",
                        },
                    ),
                    dcc.Dropdown(
                        id=dropdown_id,
                        options=[str(option) for option in options],
                        value=str(value),
                        clearable=False,
                        style={
                            "width": 50,
                            "margin": "0px auto 5px 0px",
                            "height": height - 5,
                        },
                    ),
                ],
                style={
                    "display": "table",
                    "marginLeft": 10,
                    "marginBottom": 0,
                    "marginTop": 0,
                    "height": height,
                },
            )
        )
    return dropdown_ids, dropdown_divs


def init_tulca_result_other_modes(barchart_height):
    chart_ids = [
        "factor-chart1-x",
        "factor-chart1-y",
        "factor-chart2-x",
        "factor-chart2-y",
    ]
    chart_divs = []
    for chart_id in chart_ids:
        chart_divs.append(
            html.Div(
                dcc.Graph(
                    id=chart_id, style={"height": barchart_height, "marginBottom": 5}
                )
            )
        )
    return chart_ids, chart_divs


def dict_to_css(style_dict, selector="body"):
    css = f"{selector} {{"
    for k, v in style_dict.items():
        css += f"{k}: {v}px; " if isinstance(v, (int, float)) else f"{k}: {v}; "
    css += "}"
    return css


def init_content(
    slider_divs,
    ys,
    n_insts,
    target_mode=0,
    n_components_list=[5, 5, 5],
    max_n_components_list=[20, 20, 20],
    n_rankone_tensors=3,
    show_aux_view=False,
    style_body={"font-size": 10, "background-color": "#ffffff"},  # for CSS
    style_view={  # for PlotLy
        "title": {
            "backgroundColor": "#386D8D",
            "color": "#ffffff",
            "fontWeight": "bold",
            "fontSize": 10,
            "padding": "1px 2px 1px 2px",
            "whiteSpace": "nowrap",
            "display": "inline-block",
            "width": "auto",
            "height": 14,
            "marginBottom": 0,
        },
        "default_layout": {
            "border": "1px solid #bbbbbb",
            "marginTop": 5,
            "marginRight": 0,
            "marginBottom": 0,
            "marginLeft": 5,
            "height": "97vh",
        },
        "subview_title": {
            "color": "#444444",
            "fontWeight": "bold",
            "fontSize": 10,
            "padding": "1px 2px 1px 2px",
            "whitespace": "nowrap",
            "display": "inline-block",
            "width": "auto",
            "height": 14,
            "marginBottom": 0,
        },
        "subview_layout": {
            "marginTop": 5,
            "marginRight": 0,
            "marginBottom": 0,
            "marginLeft": 0,
            "height": "40vh",
        },
    },
    tulca_result_views_width="53vh",
    proj_mat_slider1_style={"height": 350, "top": 49, "left": 243},
    proj_mat_slider2_style={"height": 275, "top": 45, "left": 243},
):
    # NOTE: style_body is to make css and needs to use "-" based description (background-color) unlike Plotly's captilized one (e.g., backgroundColor)
    css_body = dict_to_css(style_body, "body")
    style_css_path = os.path.join(os.path.dirname(__file__), "assets/style.css")

    with open(style_css_path, "w") as f:
        f.write(css_body)

    mode_radio_divs = init_mode_radios(ys=ys, target_mode=target_mode)
    _, n_components_dropdown_divs = init_n_components_drowdowns(
        n_components_list, max_n_components_list
    )
    _, scatter_dropdown_divs = init_scatterplot_axes_dropdowns(n_rankone_tensors)

    tulca_main_mode_height = tulca_result_views_width
    tulca_scatterplot_width = f"calc({tulca_result_views_width} - 3vh)"
    tulca_scatterplot_height = tulca_scatterplot_width
    tulca_other_mode_view_height = f"calc({style_view["default_layout"]["height"]} - {tulca_main_mode_height} - {style_view["default_layout"]["marginTop"]}px)"
    tulca_other_mode_barchart_height = f"calc(({style_view["default_layout"]["height"]} - {tulca_main_mode_height} - {style_view["default_layout"]["marginTop"]}px - 40px) / 4)"
    _, tulca_other_mode_divs = init_tulca_result_other_modes(
        tulca_other_mode_barchart_height
    )

    proj_mat1_height = tulca_scatterplot_height
    proj_mat2_height = f"calc({style_view["default_layout"]["height"]} - {tulca_main_mode_height} - 3vh)"

    param_view = dbc.Col(
        [
            dbc.Row(html.H5("TULCA parameters", style=style_view["title"])),
            dbc.Row(html.Div(id="hidden-div", style={"display": "none"})),
            dbc.Row(
                [
                    dbc.Col(mode_radio_divs, style={"marginTop": 10}),
                    dbc.Col(n_components_dropdown_divs, style={"marginTop": 10}),
                ]
            ),
            dbc.Row(
                slider_divs,
                style={
                    "overflowY": "scroll",
                    "maxHeight": "550px",
                    "width": "auto",
                    "marginTop": 20,
                },
            ),
            dbc.Row(scatter_dropdown_divs, style={"marginTop": 20}),
        ],
        width=2 if show_aux_view else 3,
        style=style_view["default_layout"],
    )

    tulca_result_views = dbc.Col(
        [
            dbc.Row(
                [
                    html.H5("TULCA result: Selected mode", style=style_view["title"]),
                    dcc.Graph(
                        id="scatter-plot",
                        style={
                            "width": tulca_scatterplot_width,
                            "height": tulca_scatterplot_height,
                            "marginLeft": "1.5vh",
                        },
                    ),
                ],
                style={
                    **style_view["default_layout"],
                    "width": tulca_result_views_width,
                    "height": tulca_main_mode_height,
                },
            ),
            dbc.Row(
                [
                    html.H5("TULCA result: Other modes", style=style_view["title"]),
                    *tulca_other_mode_divs,
                ],
                style={
                    **style_view["default_layout"],
                    "width": tulca_result_views_width,
                    "height": tulca_other_mode_view_height,
                },
            ),
        ],
        width="auto",
        style={"paddingRight": 0},
    )

    projection_matrix_views = dbc.Col(
        [
            dbc.Row(
                [
                    html.H5(
                        "Projection matrix 1",
                        id="proj-mat1-title",
                        style=style_view["title"],
                    ),
                    dcc.Graph(id="proj-mat1", style={"height": proj_mat1_height}),
                    html.Div(
                        [
                            dcc.RangeSlider(
                                id="colorbar-slider1",
                                className="colorbar-slider",
                                min=-1,
                                max=1,
                                step=0.01,
                                value=[-0.02, 0.02],
                                marks=None,
                                vertical=True,
                                verticalHeight=proj_mat_slider1_style["height"],
                                tooltip=None,
                            ),
                        ],
                        style={
                            "position": "absolute",
                            "top": proj_mat_slider1_style["top"],
                            "left": proj_mat_slider1_style["left"],
                        },
                    ),
                ],
                style={
                    **style_view["default_layout"],
                    "height": tulca_main_mode_height,
                    "position": "relative",
                },
            ),
            dbc.Row(
                [
                    html.H5(
                        "Projection matrix 2",
                        id="proj-mat2-title",
                        style=style_view["title"],
                    ),
                    dcc.Graph(id="proj-mat2", style={"height": proj_mat2_height}),
                    html.Div(
                        [
                            dcc.RangeSlider(
                                id="colorbar-slider2",
                                className="colorbar-slider",
                                min=-1,
                                max=1,
                                step=0.01,
                                value=[-0.02, 0.02],
                                marks=None,
                                vertical=True,
                                verticalHeight=proj_mat_slider2_style["height"],  # TODO
                                tooltip=None,
                            ),
                        ],
                        style={
                            "position": "absolute",
                            "top": proj_mat_slider2_style["top"],
                            "left": proj_mat_slider2_style["left"],
                        },  # TODO
                    ),
                ],
                style={
                    **style_view["default_layout"],
                    "height": tulca_other_mode_view_height,
                    "position": "relative",
                },
            ),
        ],
        width=3 if show_aux_view else 3,
        style={"paddingLeft": 0},
    )

    if not show_aux_view:
        content = dbc.Row([param_view, tulca_result_views, projection_matrix_views])
        return content

    subview_plot_height = f"calc({style_view["subview_layout"]["height"]} - 20px)"
    subview_lineplot_height = f"calc(15vh - 20px)"
    aux_views = dbc.Col(
        [
            dbc.Row(html.H5("Supplementary plots", style=style_view["title"])),
            dbc.Row(
                [
                    html.H5("Variable selection", style=style_view["subview_title"]),
                    dcc.Dropdown(
                        id="var-selection",
                        options=(
                            element_names_list[2]
                            if element_names_list[2] is not None
                            else [
                                {"label": str(i), "value": str(i)}
                                for i in range(1, n_insts + 1)
                            ]
                        ),
                        value=element_names_list[2][0],
                        clearable=False,
                        style={"width": "100%", "height": 20},
                    ),
                ],
                style={**style_view["subview_layout"], "height": 35},
            ),
            dbc.Row(
                [
                    html.H5(
                        "Currently selected time points (T1)",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="currently-selected-time-plot",
                        style={"height": subview_lineplot_height},
                    ),
                ],
                style={**style_view["subview_layout"], "height": "15vh"},
            ),
            dbc.Row(
                [
                    html.H5(
                        "Previously selected timepoints (T2)",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="previously-selected-time-plot",
                        style={"height": subview_lineplot_height},
                    ),
                ],
                style={**style_view["subview_layout"], "height": "15vh"},
            ),
            dbc.Row(
                [
                    html.H5(
                        "T1's mean variable value",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="currently-selected-heatmap",
                        style={"height": subview_plot_height},
                    ),
                ],
                style=style_view["subview_layout"],
            ),
            dbc.Row(
                [
                    html.H5(
                        "T2's mean variable value",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="previously-selected-heatmap",
                        style={"height": subview_plot_height},
                    ),
                ],
                style=style_view["subview_layout"],
            ),
            dbc.Row(
                [
                    html.H5(
                        "Difference between T1 and T2's mean variable values",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="diff-heatmap",
                        style={"height": subview_plot_height},
                    ),
                ],
                style=style_view["subview_layout"],
            ),
            dbc.Row(
                [
                    html.H5(
                        "Difference between T1 and all timepoints' mean variable values",
                        style=style_view["subview_title"],
                    ),
                    dcc.Graph(
                        id="diff-from-all-heatmap",
                        style={"height": subview_plot_height},
                    ),
                ],
                style=style_view["subview_layout"],
            ),
            dbc.Row(
                [
                    html.Button(
                        "Output", id="Output", n_clicks=0, style={"width": "70%"}
                    ),
                    html.Div(id="selected-data-output"),
                ],
                style={**style_view["subview_layout"], "height": 30, "marginLeft": 10},
            ),
        ],
        width=3,
        style={**style_view["default_layout"], "overflowY": "scroll"},
    )

    content = dbc.Row(
        [param_view, tulca_result_views, projection_matrix_views, aux_views]
    )
    return content


def init_app(
    X,
    ys,
    default_w_tg=None,
    default_w_bg=None,
    default_w_bw=None,
    default_target_mode=0,
    default_n_components_list=None,
    default_cp_rank=3,
    y_names_list=[None, None, None],
    element_names_list=[None, None, None],
    show_aux_view=False,
    inst_heatmap_shape=(36, 24),
    bool_heatmap_plot=True,
    proj_mat_slider1_style={"height": 350, "top": 49, "left": 243},
    proj_mat_slider2_style={"height": 275, "top": 45, "left": 243},
):
    # TODO: currently, projection matrices' sliders positions are manually set
    # but they should be set dynamically based on the projection matrix's height
    # However, it is not easy to do so with Dash

    mode_names = ["Time", "Instance", "Variable"]

    # NOTE: variables used for closures are indicated by underscore for now
    # data related
    _n_classes_list = [len(np.unique(y)) for y in ys]

    # algorithm related
    _tulca = None
    _target_mode = default_target_mode
    _n_components_list = (
        default_n_components_list
        if default_n_components_list is not None
        else np.vstack((np.array([5, 5, 5]), X.shape)).min(axis=0)
    )
    _cp_rank = default_cp_rank
    _Us = None

    # if None, set params performing TLDA
    w_tg = (
        default_w_tg
        if default_w_tg is not None
        else np.zeros(_n_classes_list[_target_mode])
    )
    w_bg = (
        default_w_bg
        if default_w_bg is not None
        else np.ones(_n_classes_list[_target_mode])
    )
    w_bw = (
        default_w_bw
        if default_w_bw is not None
        else np.ones(_n_classes_list[_target_mode])
    )

    # _weight_checks stores each weight type's classes are checked or no
    _weight_checks = np.array(
        [
            np.zeros_like(w_tg, dtype=bool),
            np.zeros_like(w_bg, dtype=bool),
            np.zeros_like(w_bw, dtype=bool),
        ]
    )

    # visualization related
    _prev_selected_indices_before = []
    _colors_list = [
        colors10 if n_classes <= 10 else colors20 for n_classes in _n_classes_list
    ]

    # slider setting  #TODO: slider should be changed dynamiacally too
    slider_ids, slider_divs = init_weight_sliders(
        w_tg,
        w_bg,
        w_bw,
        colors=_colors_list[_target_mode],
        label_names=y_names_list[_target_mode],
    )

    # algorithm setting
    _tulca = TULCA(
        n_components=np.delete(_n_components_list, _target_mode),
        w_tg=w_tg,
        w_bg=w_bg,
        w_bw=w_bw,
        optimization_method="evd",
    )
    _tulca = _tulca.fit(np.moveaxis(X, _target_mode, 0), ys[_target_mode])

    content = init_content(
        slider_divs=slider_divs,
        ys=ys,
        target_mode=_target_mode,
        n_components_list=_n_components_list,
        max_n_components_list=X.shape,
        n_insts=X.shape[1],
        show_aux_view=show_aux_view,
        proj_mat_slider1_style=proj_mat_slider1_style,
        proj_mat_slider2_style=proj_mat_slider2_style,
    )

    app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

    app.layout = dbc.Container(
        [dbc.Row([dbc.Col(content, width=12, style={"backgroundColor": "#ffffff"})])],
        fluid=True,
    )

    @app.callback(
        [
            Output("ncomps-time-dropdown", "disabled"),
            Output("ncomps-inst-dropdown", "disabled"),
            Output("ncomps-var-dropdown", "disabled"),
        ],
        [Input("mode-radios", "value")],
    )
    def update_dropdown_disabled(target_mode):
        disabled = [False, False, False]
        disabled[target_mode] = True
        return disabled

    # slider check boxes related callback (their ids are slider_id + "_checkbox")
    @app.callback(
        [],
        [Input(slider_id + "_checkbox", "value") for slider_id in slider_ids],
    )
    def update_tulca_results(
        *tulca_weight_check_values,
    ):
        nonlocal _n_classes_list, _weight_checks
        checks = [
            (val is not None) and (len(val) > 0) for val in tulca_weight_check_values
        ]
        n_classes = _n_classes_list[_target_mode]
        _weight_checks[0] = checks[:n_classes]
        _weight_checks[1] = checks[n_classes : n_classes * 2]
        _weight_checks[2] = checks[n_classes * 2 : n_classes * 3]

        return

    # NOTE: if we set updatemode=drag for slider, we can do realtime updates
    # but it calls TULCA as well and reduces the interactivity
    @app.callback(
        [Output(slider_id, "value") for slider_id in slider_ids],
        [Input(slider_id, "value") for slider_id in slider_ids],
    )
    def sync_sliders(*new_weights):
        nonlocal _tulca, _weight_checks
        old_weights = np.array([_tulca.w_tg, _tulca.w_bg, _tulca.w_bw])

        n_classes = _weight_checks.shape[1]
        new_w_tg = new_weights[:n_classes]
        new_w_bg = new_weights[n_classes : n_classes * 2]
        new_w_bw = new_weights[n_classes * 2 : n_classes * 3]
        new_weights = np.array([new_w_tg, new_w_bg, new_w_bw])

        # check which weight is updated
        updates = np.abs(new_weights - old_weights) > 0

        # if updated weight has synced weights, update them too
        for index in np.argwhere(updates > 0):
            updated_val = new_weights[index[0], index[1]]
            if _weight_checks[index[0], index[1]] > 0:
                new_weights[index[0], _weight_checks[index[0]]] = updated_val

        return tuple(new_weights.flatten())

    @app.callback(
        [
            Output("scatter-plot", "figure"),
            Output("factor-chart1-x", "figure"),
            Output("factor-chart1-y", "figure"),
            Output("factor-chart2-x", "figure"),
            Output("factor-chart2-y", "figure"),
            Output("proj-mat1", "figure"),
            Output("proj-mat2", "figure"),
            Output("colorbar-slider1", "value"),
            Output("colorbar-slider2", "value"),
        ],
        [
            Input("mode-radios", "value"),
            Input("ncomps-time-dropdown", "value"),
            Input("ncomps-inst-dropdown", "value"),
            Input("ncomps-var-dropdown", "value"),
            Input("xaxis-factor-index-dropdown", "value"),
            Input("yaxis-factor-index-dropdown", "value"),
        ]
        + [Input(slider_id, "value") for slider_id in slider_ids],
    )
    def update_tulca_results(
        target_mode,
        n_components_time,
        n_components_inst,
        n_components_var,
        xaxis_factor_index_dropdown_value,
        yaxis_factor_index_dropdown_value,
        *tulca_weights,
    ):
        nonlocal _tulca, _target_mode, _n_components_list, _Us

        prev_target_mode = _target_mode
        _target_mode = target_mode

        prev_n_components_list = _n_components_list.copy()
        _n_components_list = np.array(
            [int(n_components_time), int(n_components_inst), int(n_components_var)]
        )

        n_classes = _n_classes_list[_target_mode]
        w_tg = tulca_weights[:n_classes]
        w_bg = tulca_weights[n_classes : n_classes * 2]
        w_bw = tulca_weights[n_classes * 2 : n_classes * 3]
        # TODO: we don't use alpha now
        # alpha = tulca_weights[-1]

        # X_rotated will be when _target_mode=0: TxNxD, 1: NxTxD, 2: DxTxN
        X_rotated = np.moveaxis(X, _target_mode, 0)

        # TULCA
        if (prev_target_mode == _target_mode) and (
            np.array_equal(prev_n_components_list, _n_components_list)
        ):
            # can reuse the fit result with new weights
            _tulca = _tulca.fit_with_new_weights(w_tg=w_tg, w_bg=w_bg, w_bw=w_bw)
        else:
            # need to update everything
            # TODO: we can compute all mode in advance though
            _tulca = TULCA(
                n_components=np.delete(_n_components_list, _target_mode),
                optimization_method="evd",
            )
            _tulca = _tulca.fit(X_rotated, ys[_target_mode])
        X_tulca = _tulca.transform(X_rotated)

        # CP decompositon
        cp_weights, cp_factors = tl.decomposition.parafac(X_tulca, rank=_cp_rank)
        _Us = _tulca.get_projection_matrices()

        # reordering
        for i in range(len(_Us)):
            _Us[i], col_order, row_order = reorder_matrix_by_similarity(_Us[i])
            # TULCA's Us correspond modes from the second (e.g., TxNxD => N and D's)
            cp_factors[i + 1] = cp_factors[i + 1][col_order]
        xaxis_factor_index = int(xaxis_factor_index_dropdown_value) - 1
        yaxis_factor_index = int(yaxis_factor_index_dropdown_value) - 1
        mode_names = ["Time", "Instance", "Variable"]
        other_mode_indices = np.delete([0, 1, 2], _target_mode)
        other_mode_name1 = mode_names[other_mode_indices[0]]
        other_mode_name2 = mode_names[other_mode_indices[1]]

        return (
            cp_factor_scatter(
                cp_factors,
                labels=ys[_target_mode],
                mode=0,
                x_factor_index=xaxis_factor_index,
                y_factor_index=yaxis_factor_index,
                colors=_colors_list[_target_mode],
                label_names=y_names_list[_target_mode],
            ),
            cp_factor_bar(
                cp_factors,
                1,
                xaxis_factor_index,
                f"{other_mode_name1} component",
                "x-axis-value",
                True,
            ),
            cp_factor_bar(
                cp_factors,
                1,
                yaxis_factor_index,
                f"{other_mode_name1} component",
                "y-axis-value",
                False,
            ),
            cp_factor_bar(
                cp_factors,
                2,
                xaxis_factor_index,
                f"{other_mode_name2} component",
                "x-axis-value",
                True,
            ),
            cp_factor_bar(
                cp_factors,
                2,
                yaxis_factor_index,
                f"{other_mode_name2} component",
                "y-axis-value",
                False,
            ),
            heatmap(
                _Us[0],
                y_ticks=element_names_list[other_mode_indices[0]],
                xlabel=f"{other_mode_name1} component",
                ylabel=other_mode_name1,
                colorbar_legend="Loadings",
            ),
            heatmap(
                _Us[1],
                y_ticks=element_names_list[other_mode_indices[1]],
                xlabel=f"{other_mode_name2} component",
                ylabel=other_mode_name2,
                colorbar_legend="Loadings",
            ),
            [-0.02, 0.02],
            [-0.02, 0.02],
        )

    if show_aux_view:
        selected_data_decorator = app.callback(
            [
                Output("currently-selected-time-plot", "figure"),
                Output("previously-selected-time-plot", "figure"),
                Output("currently-selected-heatmap", "figure"),
                Output("previously-selected-heatmap", "figure"),
                Output("diff-heatmap", "figure"),
                Output("diff-from-all-heatmap", "figure"),
            ],
            [
                Input("scatter-plot", "selectedData"),
                Input("var-selection", "value"),
            ],
        )
    else:
        selected_data_decorator = app.callback(
            [], [Input("scatter-plot", "selectedData")]
        )

    @selected_data_decorator
    def display_selected_data(selected_data, selected_var=None):
        global states
        nonlocal _prev_selected_indices_before

        class_start_indices = np.insert(
            np.cumsum(np.unique(ys[_target_mode], return_counts=True)[1]), 0, 0
        )
        if selected_data is not None:
            points = selected_data["points"]
            selected_indices = []
            for point in points:
                # curveNumber corresponds to the class index
                # pointIndex corresponds to the appeared order in the class
                label = point["curveNumber"]
                selected_indices.append(
                    np.where(ys[_target_mode] == label)[0][point["pointIndex"]]
                )
            selected_indices = np.array(selected_indices)

            states["selected_indices"] = selected_indices

        if not show_aux_view:
            return
        else:
            if selected_data is None:
                return [
                    {
                        "data": [],
                        "layout": {
                            "margin": dict(t=50, b=50, l=50, r=50),
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                        },
                    }
                    for _ in range(6)
                ]

            if element_names_list[2] is not None:
                selected_var_index = list(element_names_list[2]).index(selected_var)
            else:
                selected_var_index = int(selected_var) - 1

            mean_data = np.mean(X[:, :, selected_var_index], axis=1)
            curr_highlights = np.full_like(mean_data, np.nan)
            curr_highlights[selected_indices] = mean_data[selected_indices]

            mean_line = go.Scatter(
                x=np.arange(mean_data.shape[0]),
                y=mean_data,
                mode="lines",
                name="Average Data",
            )
            highlighted_points = go.Scatter(
                x=np.arange(mean_data.shape[0]),
                y=curr_highlights,
                mode="markers",
                marker=dict(size=8, color="black"),
                name="Selected Points",
            )

            mean_line.update(line=dict(color="gray"))
            highlighted_points.update(line=dict(color="black"))

            layout = go.Layout(
                plot_bgcolor="#ffffff",
                xaxis=dict(
                    visible=True,
                    title=dict(text="Time", font=dict(size=10)),
                    showgrid=True,
                    zeroline=True,
                    gridcolor="#888888",
                    zerolinecolor="#888888",
                    zerolinewidth=1,
                    tickfont=dict(size=10),
                ),
                yaxis=dict(
                    visible=True,
                    title=dict(text="Value", font=dict(size=10)),
                    showticklabels=True,
                    showgrid=False,
                    zeroline=True,
                    zerolinecolor="#888888",
                    zerolinewidth=1,
                    tickfont=dict(size=10),
                ),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )

            current_time_lineplot = go.Figure(
                data=[mean_line, highlighted_points], layout=layout
            )

            heatmap_layout = {
                "xaxis": dict(
                    tickfont=dict(size=10),
                    title=dict(text="Rack X-coordinate", font=dict(size=10)),
                ),
                "yaxis": dict(
                    tickfont=dict(size=10),
                    title=dict(text="Rack Y-coordinate", font=dict(size=10)),
                ),
                "coloraxis": dict(
                    colorscale="Tropic",
                    cmin=-2.0,
                    cmax=2.0,
                    colorbar_title="Value",
                    colorbar_title_font=dict(size=10),
                    colorbar_tickfont=dict(size=10),
                ),
                "margin": dict(t=10, b=10, l=10, r=10),
            }

            if bool_heatmap_plot:
                Z_curr_heatmap = np.zeros((X.shape[1]), dtype=float).reshape(
                    inst_heatmap_shape
                )
                for i in selected_indices:
                    heatmap_slice = X[i, :, selected_var_index].reshape(
                        inst_heatmap_shape
                    )
                    Z_curr_heatmap += heatmap_slice

                Z_curr_heatmap /= len(selected_indices)
                current_heatmap = go.Figure()
                current_heatmap.add_trace(
                    go.Heatmap(
                        z=Z_curr_heatmap,
                        x=[i for i in range(Z_curr_heatmap.shape[1])],
                        y=[i for i in range(Z_curr_heatmap.shape[0])],
                        coloraxis="coloraxis",
                    )
                )
                current_heatmap.update_layout(**heatmap_layout)

                if (_prev_selected_indices_before is None) or (
                    len(_prev_selected_indices_before) == 0
                ):
                    previous_heatmap = {
                        "data": [],
                        "layout": {
                            "margin": dict(t=50, b=50, l=50, r=50),
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                        },
                    }
                    previous_time_lineplot = {
                        "data": [],
                        "layout": {
                            "margin": dict(t=50, b=50, l=50, r=50),
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                        },
                    }
                    diff_heatmap = {
                        "data": [],
                        "layout": {
                            "margin": dict(t=50, b=50, l=50, r=50),
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                        },
                    }
                    diff_from_all_heatmap = {
                        "data": [],
                        "layout": {
                            "margin": dict(t=50, b=50, l=50, r=50),
                            "xaxis": {"visible": False},
                            "yaxis": {"visible": False},
                        },
                    }
                else:
                    Z_prev_heatmap = np.zeros((X.shape[1]), dtype=float).reshape(
                        inst_heatmap_shape
                    )
                    for i in _prev_selected_indices_before:
                        heatmap_slice = X[i, :, selected_var_index].reshape(
                            inst_heatmap_shape
                        )
                        Z_prev_heatmap += heatmap_slice

                    Z_prev_heatmap /= len(_prev_selected_indices_before)
                    previous_heatmap = go.Figure()
                    previous_heatmap.add_trace(
                        go.Heatmap(
                            z=Z_prev_heatmap,
                            x=[i for i in range(Z_prev_heatmap.shape[1])],
                            y=[i for i in range(Z_prev_heatmap.shape[0])],
                            coloraxis="coloraxis",
                        )
                    )
                    previous_heatmap.update_layout(**heatmap_layout)

                    Z_heatmap_diff = Z_curr_heatmap - Z_prev_heatmap
                    diff_heatmap = go.Figure()
                    diff_heatmap.add_trace(
                        go.Heatmap(
                            z=Z_heatmap_diff,
                            x=[i for i in range(Z_heatmap_diff.shape[1])],
                            y=[i for i in range(Z_heatmap_diff.shape[0])],
                            coloraxis="coloraxis",
                        )
                    )
                    diff_heatmap.update_layout(**heatmap_layout)

                    Z_heatmap_all = np.zeros((X.shape[1]), dtype=float).reshape(
                        inst_heatmap_shape
                    )
                    for i in range(len(X)):
                        heatmap_slice = X[i, :, selected_var_index].reshape(
                            inst_heatmap_shape
                        )
                        Z_heatmap_all += heatmap_slice

                    Z_heatmap_all /= len(X)
                    Z_heatmap_diff_all = Z_curr_heatmap - Z_heatmap_all
                    diff_from_all_heatmap = go.Figure()
                    diff_from_all_heatmap.add_trace(
                        go.Heatmap(
                            z=Z_heatmap_diff_all,
                            x=[i for i in range(Z_heatmap_diff_all.shape[1])],
                            y=[i for i in range(Z_heatmap_diff_all.shape[0])],
                            coloraxis="coloraxis",
                        )
                    )
                    diff_from_all_heatmap.update_layout(**heatmap_layout)

                    prev_highlights = np.full_like(mean_data, np.nan)
                    prev_highlights[_prev_selected_indices_before] = mean_data[
                        _prev_selected_indices_before
                    ]
                    mean_line = go.Scatter(
                        x=np.arange(mean_data.shape[0]),
                        y=mean_data,
                        mode="lines",
                        name="Average Data",
                    )
                    highlighted_points = go.Scatter(
                        x=np.arange(mean_data.shape[0]),
                        y=prev_highlights,
                        mode="markers",
                        marker=dict(size=8, color="black"),
                        name="Selected Points",
                    )

                    mean_line.update(line=dict(color="gray"))
                    highlighted_points.update(line=dict(color="black"))
                    previous_time_lineplot = go.Figure(
                        data=[mean_line, highlighted_points], layout=layout
                    )
            else:
                Z_curr_heatmap = np.zeros((X.shape[1]), dtype=float)
                for i in selected_indices:
                    heatmap_slice = X[i, :, selected_var_index]
                    Z_curr_heatmap += heatmap_slice

                Z_curr_heatmap /= len(selected_indices)
                current_heatmap = go.Figure()
                current_heatmap.add_trace(
                    go.Scatter(
                        x=[i for i in range(len(Z_curr_heatmap))],
                        y=Z_curr_heatmap,
                        mode="lines",
                        name="Average Data",
                    )
                )
                current_heatmap.update_layout(**heatmap_layout)

            _prev_selected_indices_before = selected_indices

            return (
                current_time_lineplot,
                previous_time_lineplot,
                current_heatmap,
                previous_heatmap,
                diff_heatmap,
                diff_from_all_heatmap,
            )

    @app.callback(
        Output("proj-mat1", "figure", allow_duplicate=True),
        Input("colorbar-slider1", "value"),
        prevent_initial_call=True,
    )
    def filter_projection_matrix1(slider_range):
        nonlocal _tulca, _target_mode, _Us

        other_mode_indices = np.delete([0, 1, 2], _target_mode)
        other_mode_name1 = mode_names[other_mode_indices[0]]

        # slider_range is [-1, 1] and coverts to [-abs_max, max]
        filter_out_range = np.array(slider_range) * np.abs(_Us[0]).max()

        return heatmap(
            _Us[0],
            y_ticks=element_names_list[other_mode_indices[0]],
            xlabel=f"{other_mode_name1} component",
            ylabel=other_mode_name1,
            colorbar_legend="Loadings",
            filter_out_range=filter_out_range,
        )

    @app.callback(
        Output("proj-mat2", "figure", allow_duplicate=True),
        Input("colorbar-slider2", "value"),
        prevent_initial_call=True,
    )
    def filter_projection_matrix2(slider_range):
        nonlocal _tulca, _target_mode, _Us

        other_mode_indices = np.delete([0, 1, 2], _target_mode)
        other_mode_name2 = mode_names[other_mode_indices[1]]

        # slider_range is [-1, 1] and coverts to [-abs_max, max]
        filter_out_range = np.array(slider_range) * np.abs(_Us[1]).max()

        return heatmap(
            _Us[1],
            y_ticks=element_names_list[other_mode_indices[1]],
            xlabel=f"{other_mode_name2} component",
            ylabel=other_mode_name2,
            colorbar_legend="Loadings",
            filter_out_range=filter_out_range,
        )

    app.title = "TULCA UI"
    return app


if __name__ == "__main__":
    debug_mode = False

    # TODO: currently, projection matrices' sliders positions are manually set
    # Adjust them to fit your colormap height and position
    proj_mat_slider1_style = {"height": 350, "top": 49, "left": 257}
    proj_mat_slider2_style = {"height": 275, "top": 45, "left": 257}

    analysis_case = "highschool_inst"
    # analysis_case = "mhealth"

    # K log dataset is not publicly available
    # analysis_case = "klog_time"
    # analysis_case = "klog_inst"
    # if analysis_case == "klog_time" or analysis_case == "klog_inst":
    #     X = np.load("../data/k_log/tensor.npy")

    #     y_time = np.load("../data/k_log/time_labels.npy")
    #     y_inst = np.random.choice(10, X.shape[1])  # dummy example for test
    #     ys = [y_time, y_inst, None]
    #     if analysis_case == "klog_time":
    #         y_names_list = [["FY2014", "FY2015", "FY2016"], None, None]
    #     element_names_list = [None, None, ["AirIn", "AirOut", "CPU", "Water"]]

    #     target_mode = 0 if analysis_case == "klog_time" else 1
    #     app = init_app(
    #         X,
    #         ys,
    #         y_names_list=y_names_list,
    #         element_names_list=element_names_list,
    #         default_target_mode=target_mode,  # 0: time, 1: instance, 2: varible
    #         default_n_components_list=np.array([10, 10, 4]),
    #         show_aux_view=True,
    #         proj_mat_slider1_style=proj_mat_slider1_style,
    #         proj_mat_slider2_style=proj_mat_slider2_style,
    #     )
    #     app.run(debug=debug_mode)
    # elif analysis_case == "highschool_inst":
    if analysis_case == "highschool_inst":
        import pandas as pd

        X = np.load("../data/highschool_2012/tensor.npy")

        label_df = pd.read_csv("../data/highschool_2012/instance_labels.csv")
        y_inst = np.array(label_df["label"])
        y_inst_names = np.array(label_df.groupby("label")["label_name"].first())

        time_names = pd.read_csv("../data/highschool_2012/times.csv")["check_time"]
        var_names = pd.read_csv("../data/highschool_2012/variables.csv")["name"]

        ys = [None, y_inst, None]
        y_names_list = [None, y_inst_names, None]
        element_names_list = [time_names, None, var_names]

        app = init_app(
            X,
            ys,
            y_names_list=y_names_list,
            element_names_list=element_names_list,
            default_target_mode=1,  # 0: time, 1: instance, 2: varible
            default_n_components_list=np.array([5, 10, 3]),
            proj_mat_slider1_style=proj_mat_slider1_style,
            proj_mat_slider2_style=proj_mat_slider2_style,
        )

        app.run(debug=debug_mode)

    elif analysis_case == "mhealth":
        import pandas as pd

        X = np.load("../data/mhealth/tensor.npy")
        n_classes = 8
        y_time = np.load("../data/mhealth/time_labels.npy")
        y_time_names = np.array(
            [
                "Standing still",
                "Sitting and relaxing",
                "Lying down",
                "Walking",
                "Climbing stairs",
                "Cycling",
                "Jogging",
                "Running",
            ]
        )

        var_names = np.array(
            [
                "Accel chest (x)",
                "Accel chest (y)",
                "Accel chest (z)",
                "ECG signal (1)",
                "ECG signal (2)",
                "Accel L-ankle (x)",
                "Accel L-ankle (y)",
                "Accel L-ankle (z)",
                "Gyro L-ankle (x)",
                "Gyro L-ankle (y)",
                "Gyro L-ankle (z)",
                "Magnet L-ankle (x)",
                "Magnet L-ankle (y)",
                "Magnet L-ankle (z)",
                "Accel R-arm (x)",
                "Accel R-arm (y)",
                "Accel R-arm (z)",
                "Gyro R-arm (x)",
                "Gyro R-arm (y)",
                "Gyro R-arm (z)",
                "Magnet R-arm (x)",
                "Magnet R-arm (y)",
                "Magnet R-arm (z)",
            ]
        )

        ys = [y_time, None, None]
        y_names_list = [y_time_names, None, None]
        element_names_list = [None, None, var_names]

        app = init_app(
            X,
            ys,
            y_names_list=y_names_list,
            element_names_list=element_names_list,
            default_target_mode=0,  # 0: time, 1: instance, 2: varible
            default_n_components_list=np.array([10, 3, 4]),
            proj_mat_slider1_style=proj_mat_slider1_style,
            proj_mat_slider2_style=proj_mat_slider2_style,
        )

        app.run(debug=debug_mode)
