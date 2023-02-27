from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats, special

class bounded_beta(stats.rv_continuous):
    def _pdf(self, x, alpha, beta):
        mask = np.greater(x,self.a) & np.less(x,self.b)
        y = np.zeros_like(x)
        # this is the formula for pdf of beta distribution
        # note the confusion about the beta function
        # it's different from the beta parameter
        y[mask] = x[mask]**(alpha[mask]-1) * (1-x[mask])**(beta[mask]-1) / special.beta(alpha[mask], beta[mask])
        return y

input_beta = html.Div(
    [
        html.Div('Lower bound'),
        dcc.Slider(
            id="lb",
            min=0,
            max=1,
            value=0.1,
            step = 0.1,
            className="four columns",
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div('Upper bound'),
        dcc.Slider(
            id="ub",
            min=0,
            max=1,
            value=0.8,
            step = 0.1,
            className="four columns",
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div('Mode'),
        dcc.Slider(
            id="mode",
            min=0,
            max=1,
            value=0.5,
            step = 0.05,
            className="four columns",
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div('Concentration (alpha + beta)'),
        dcc.Slider(
            id="conc",
            min=0,
            max=400,
            step=1,
            value=0,
            className="four columns",
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        # html.H4("Distribution parameters"),
        # html.Div(id="beta_mean"),   # these can't be inputs, since their existence depends on whether beta is chosen or not
        # html.Div(id="beta_sd"),     # need to reformulate in another way
        # html.Div(id="beta_median"),
    ],
    className="row",
    id="input_beta",
)

input_skew = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("Alpha (skew)"),
                dbc.Input(
                    id="alpha_skew",
                    placeholder = 0,
                    type="number",
                    value=0
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Location"),
                dbc.Input(
                    id="mean_skew",
                    placeholder = 0,
                    type="number",
                    value=0
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Scale"),
                dbc.Input(
                    id="std_skew",
                    placeholder = 1,
                    type="number",
                    value=1
                ),
            ],
            className="mb-3",
        )
    ],
    className="row",
    id="input_skew",
)

input_mv = html.Div(
    [
        dbc.InputGroup(
            [
                dbc.InputGroupText("1st variable mean"),
                dbc.Input(
                    id="var1_mu",
                    placeholder="10",
                    type="number",
                    min=10,
                    value=58.8,
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("1st variable standard deviation"),
                dbc.Input(
                    id="var1_sd",
                    placeholder="3",
                    type="number",
                    min=0.01,
                    value=7.82,
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("2nd variable mean"),
                dbc.Input(
                    id="var2_mu",
                    placeholder="7",
                    type="number",
                    min=10,
                    value=60.7,
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("2nd variable standard deviation"),
                dbc.Input(
                    id="var2_sd",
                    placeholder="10",
                    type="number",
                    min=0.01,
                    value=7.6,
                ),
            ],
            className="mb-3",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Variable correlation"),
                dbc.Input(
                    id="corr",
                    placeholder="0.12",
                    type="number",
                    min=-1,
                    value=0.7,
                ),
            ],
            className="mb-3",
        ),
    ],
    className="row",
    id="input_mv",
)

# App Layout *******************************************

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Visualizing Distributions",
                    className="text-center bg-primary text-white p-2",
                ),
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            "Select distribution",
                            style={"textAlign": "center"},
                        ),
                        dcc.Dropdown(
                            id="select-distribution",
                            options=[
                                {"label": "Beta", "value": "beta"},
                                {
                                    "label": "Multivariate normal",
                                    "value": "multivariate_normal",
                                },
                                {"label": "Skew normal", "value": "skew_norm"}
                            ],
                            value="beta",
                        ),
                        html.Div(
                            [input_beta, input_mv, input_skew],
                            id="sliders-or-inputs",
                            className="mt-4 p-4",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div([], id="beta_graph"),
                        html.Div([], id="mv_graph"),
                        html.Div([], id="skew_graph")
                    ],
                    width=8,
                ),
            ]
        ),
    ]
)

# Callbacks *******************************************


@app.callback(
    Output("input_beta", "style"),
    Output("input_mv", "style"),
    Output("input_skew", "style"),
    Input("select-distribution", "value"),
)
def display_sliders_or_inputs(distribution):
    if distribution == "beta":
        return {}, {"display": "none"}, {"display": "none"}
    elif distribution == "multivariate_normal":
        return {"display": "none"}, {}, {"display", "none"}
    elif distribution =="skew_norm":
       return {"display": "none"}, {"display": "none"}, {}


@app.callback(
    Output("mv_graph", "children"),
    Input("select-distribution", "value"),
    Input("var1_mu", "value"),
    Input("var1_sd", "value"),
    Input("var2_mu", "value"),
    Input("var2_sd", "value"),
    Input("corr", "value"),
)
def update_mv_normal_distr(distribution, var1_mu, var1_sd, var2_mu, var2_sd, corr):
    if distribution == "multivariate_normal":
        x1, x2 = np.mgrid[40:80:0.25, 40:80:0.25]
        means = np.array([var1_mu, var2_mu])

        cov = var1_sd * var2_sd * corr
        cov_mat = np.array([[var1_sd**2, cov], [cov, var2_sd**2]])

        z = stats.multivariate_normal(means, cov_mat).pdf(np.dstack((x1, x2)))

        fig = go.Figure(data=[go.Surface(z=z)])
        fig.update_xaxes(title_text="Testing distribution display")
        fig.update_yaxes(title_text="Probability Density")
        fig.update_layout(
            width=700,
            height=700,
            title="Multivariate Normal Distribution",
            scene=dict(
                xaxis=dict(title="Var1"),
                yaxis=dict(title="Var2"),
                zaxis=dict(title="Probability density"),
            ),
            margin=dict(l=0, r=50, b=50, t=50),
        )

        return dcc.Graph(figure=fig)
    else:
        return None


@app.callback(
    Output("beta_mean", "children"),
    Output("beta_sd", "children"),
    Output("beta_median", "children"),
    Input("select-distribution", "value"),
    Input("alpha", "value"),
    Input("beta", "value"),
)
def get_beta_stats(distribution, a=5, b=2):
    if distribution == "beta":
        beta_smp = stats.beta(a, b).rvs(100000)
        mu = f"Mean: {round(np.mean(beta_smp), 3)}"
        sigma = f"Standard deviation: {round(np.std(beta_smp), 3)}"
        med = f"Median: {round(np.median(beta_smp), 3)}"

        return mu, sigma, med
    else:
        return None


@app.callback(
    Output("beta_graph", "children"),
    Input("select-distribution", "value"),
    Input("lb", "value"),
    Input("ub", "value"),
    Input("mode", "value"),
    Input("conc", "value"),
)
def update_beta_distr(distribution, lb, ub, m, conc):
    if distribution == "beta":

        bounded_beta_distribution = bounded_beta(a=lb, b=ub, name='bounded_beta',
                                        shapes='alpha, beta')

        alpha_ = m * (conc - 2) + 1
        beta_ = (1-m) * (conc - 2) + 1

        x = np.linspace(lb, ub, 10000)
        y = bounded_beta_distribution.pdf(x, alpha_, beta_)
        fig = px.line(x=x, y=y, title="Beta distribution")
        fig.update_layout(xaxis_title="Value", yaxis_title="Relative probability")

        return dcc.Graph(figure=fig)
    else:
        return None

@app.callback(
    Output("skew_graph", "children"),
    Input("select-distribution", "value"),
    Input("alpha_skew", "value"),
    Input("mean_skew", "value"),
    Input("std_skew", "value")
)
def update_skewnorm_distr(distribution, a, mu, std_):
    if distribution=="skew_norm":
        x = np.linspace(-5, 5, 10000)
        y = stats.skewnorm.pdf(x, a, mu, std_)
        fig = px.line(x = x, y = y, title = "Skew Normal Distribution")
        fig.update_layout(xaxis_title="Value", yaxis_title="Relative Probability")

        return dcc.Graph(figure=fig)
    else:
        return None


if __name__ == "__main__":
    app.run_server(debug=True)
