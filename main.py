import dash
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc
from layout import Layout
import data_util
import numpy as np
import json
from plotly.subplots import make_subplots
from dash_canvas.utils import parse_jsonstring

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], title="Handwritten digits classifier")
canvas_size = 300
server = app.server

layout_instance = Layout(canvas_size)
app.layout = html.Div([
    layout_instance.layout(),
    html.Div(id="reload-output", style={"display": "none"})
])

@app.callback(
    [Output("prediction-output", "children"),
     Output("prediction-plot", "figure")],
    [Input("canvas_image", "json_data")],
    prevent_initial_call=True
)
def update_prediction(json_data):
    if not json_data:
        return "No drawing detected. Please draw a digit on the canvas.", dash.no_update
    
    try:
        json_dict = json.loads(json_data)
        if "objects" in json_dict and json_dict["objects"]:
            img_array = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
            
            mask = parse_jsonstring(json_data, img_array.shape)
            
            img_array[mask > 0] = 0
            
            knn_prediction, knn_processed_image = data_util.predict_image(img_array, model_type="knn")
            knn_reshaped = knn_processed_image.reshape(28, 28)
            
            nn_prediction, nn_processed_image = data_util.predict_image(img_array, model_type="nn")
            nn_reshaped = nn_processed_image.reshape(28, 28)
            
            fig = data_util.create_plot(
                left_image=img_array, 
                right_image=knn_reshaped,
                nn_image=nn_reshaped,
                left_title="Input Drawing", 
                knn_title="KNearest Prediction",
                nn_title="Neural Network Prediction",
                knn_prediction=knn_prediction,
                nn_prediction=nn_prediction
            )
            
            prediction_text = f"KNN: {knn_prediction[0]} | Neural Network: {nn_prediction[0]}"
            return prediction_text, fig
        else:
            return "No drawing detected. Please draw a digit on the canvas.", dash.no_update
            
    except Exception as e:
        import traceback
        print(f"Error processing drawing: {str(e)}")
        print(traceback.format_exc())
        return f"Error: {str(e)}", dash.no_update

# Clientside callback that reloads the entire page, since dash doesnt allow resetting the canvas
# This is a workaround to clear the canvas
app.clientside_callback(
    """
    function(n_clicks) {
      if (n_clicks) {
        window.location.reload();
      }
      return "";
    }
    """,
    Output("reload-output", "children"),
    Input("clear-button", "n_clicks")
)

if __name__ == '__main__':
    app.run_server(debug=True)