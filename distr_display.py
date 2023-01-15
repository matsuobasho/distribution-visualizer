from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

input_beta = html.Div(
                    [
                        dcc.Slider(
                            id='alpha',
                            min=0,
                            max=100,
                            value=0,
                            className='four columns',
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        dcc.Slider(
                            id='beta',
                            min=0,
                            max=100,
                            value=0,
                            className='four columns',
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.H4("Distribution parameters"),
                        html.Div(id="beta_mean"),
                        html.Div(id="beta_sd"),
                        html.Div(id="beta_median")
                    ],
                    className='row',
                    id= "input_beta"
                )

input_mv = html.Div([
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
        ], className='row',
        id="input_mv"
        )

# App Layout *******************************************

app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container(
    [
        dbc.Row
        (
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
                    [html.H3(
                            "Select distribution",
                                style={"textAlign": "center"},
                            ),
                    dcc.Dropdown(
                                id='select-distribution',
                                options=[
                                    {'label': 'Beta', 'value': 'beta'},
                                    {'label': 'Multivariate normal', 'value': 'multivariate_normal'}
                                ],
                                value='beta',
                            ),
                    html.Div(
                        [
                            input_beta,
                            input_mv
                        ],
                        id = 'sliders-or-inputs',
                        className="mt-4 p-4"
                    )
                    ],
                    width=4
                    ),
            dbc.Col(
                    [
                    html.Div([], id='beta_graph'),
                    html.Div([], id='mv_graph'),
                    ],
                    width=8
                    ),
            ]
            )
    ]
)

# Callbacks *******************************************

@app.callback(
    Output('input_beta', 'style'),
    Output('input_mv', 'style'),
    Input('select-distribution', 'value'),

)
def display_sliders_or_inputs(distribution):
    if distribution == 'beta':
        return {},{'display': 'none'}
    elif distribution == 'multivariate_normal':
        return  {'display': 'none'},{}

@app.callback(
    Output("mv_graph", "children"),
    Input('select-distribution', 'value'),
    Input("var1_mu", "value"),
    Input("var1_sd", "value"),
    Input("var2_mu", "value"),
    Input("var2_sd", "value"),
    Input("corr", "value"),
)
def update_mv_normal_distr(distribution, var1_mu, var1_sd, var2_mu, var2_sd, corr):
    if distribution=='multivariate_normal':
        x1, x2 = np.mgrid[40:80:0.25, 40:80:0.25]
        means = np.array([var1_mu, var2_mu])

        cov = var1_sd * var2_sd * corr
        cov_mat = np.array([[var1_sd**2, cov], [cov, var2_sd**2]])

        z = stats.multivariate_normal(means, cov_mat).pdf(np.dstack((x1, x2)))

        fig = go.Figure(data=[go.Surface(z=z)])
        fig.update_xaxes(title_text='Testing distribution display')
        fig.update_yaxes(title_text='Probability Density')
        fig.update_layout(width=700, height=700, title="Multivariate Normal Distribution",
                        scene = dict(xaxis = dict(title = 'Var1'),
                                    yaxis = dict(title = 'Var2'),
                                    zaxis = dict(title = 'Probability density')),
                        margin=dict(l=0, r=50, b=50, t=50))

        return dcc.Graph(figure=fig)
    else:
        return None

@app.callback(
    Output("beta_mean", "children"),
    Output("beta_sd", "children"),
    Output("beta_median", "children"),
    Input('select-distribution', 'value'),
    Input("alpha", "value"),
    Input("beta", "value")
)
def get_beta_stats(distribution, a = 5, b = 2):
    if distribution=='beta':
        beta_smp = stats.beta(a,b).rvs(100000)
        mu = f'Mean: {round(np.mean(beta_smp), 3)}'
        sigma = f'Standard deviation: {round(np.std(beta_smp), 3)}'
        med = f'Median: {round(np.median(beta_smp), 3)}'

        return mu, sigma, med
    else:
        return None

@app.callback(
    Output("beta_graph", "children"),
    Input('select-distribution', 'value'),
    Input("alpha", "value"),
    Input("beta", "value")
    )
def update_beta_distr(distribution, a, b):
    if distribution=='beta':
        x = np.linspace(0, 1, 10000)
        beta_distr = stats.beta(a,b).pdf(x)
        fig = px.line(x = x, y = beta_distr, title="Beta distribution")
        fig.update_layout(xaxis_title='Value', yaxis_title='Relative probability')

        return dcc.Graph(figure=fig)
    else:
        return None

if __name__ == "__main__":
    app.run_server(debug=True)