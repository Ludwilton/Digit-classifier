import dash_bootstrap_components as dbc
from dash import dcc, html
from dash_canvas import DashCanvas
from data_util import create_plot
import json


class Layout:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size

    def get_new_canvas(self):
        return DashCanvas(
            id='canvas_image',
            scale=1,
            lineWidth=13,
            lineColor='black',
            width=self.canvas_size,
            height=self.canvas_size,
            hide_buttons=[
                "zoom", "pan", "line", "pencil",
                "rectangle", "undo", "select"
            ],
            goButtonTitle="Predict Digit",
            tool="pencil",
            filename="",
            json_data=json.dumps({"version": "4.4.0", "objects": [], "background": "#ffffff"}),
        )

    def layout(self):
        main_content = html.Div([
            dbc.Row([
                dbc.Col([

                    html.Div(
                        id="canvas-container",
                        children=[self.get_new_canvas()],
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "border": "0.5px solid #dee2e6",
                            "backgroundColor": "white",
                            "width": f"{self.canvas_size}px",
                            "height": f"{self.canvas_size-50}px",
                            "margin": "0 auto",
                            "padding": "0",
                            "overflow": "hidden"
                        }
                    ),

                    html.Div([
                        dbc.Button("Reset Canvas", id="clear-button", color="secondary", 
                                  className="me-2"),
                    ], className="mt-3 d-flex justify-content-center")
                ], width=12, md=6, className="mb-4 mx-auto")
            ], className="justify-content-center"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Prediction Result", className="mb-3"),
                        html.Div(id="prediction-output", className="mb-3"),
                        html.Div([
                            dcc.Graph(
                                id="prediction-plot", 
                                figure=create_plot(),
                                config={'displayModeBar': False},
                                style={'width': '100%', 'height': '400px'},
                                className="mx-auto"
                            )
                        ], style={
                            "border": "2px solid #dee2e6",
                        })
                    ], className="d-flex flex-column align-items-center")
                ], width=12, md=8, lg=6, className="mx-auto")
            ], className="justify-content-center"),
        ])



        return dbc.Container([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Handwritten Digits Classifier", style={"margin": "0"}),
                ])
            ], className="mb-3"),
            dbc.Card(dbc.CardBody(main_content), className="mt-3")
        ], fluid=True)