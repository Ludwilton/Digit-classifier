# import matplotlib
# matplotlib.use('Agg') # matplotlib must be run on main thread, workaround for debugging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress tensorflow warnings


from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize
from skimage import util
import joblib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import os


MODEL_PATH = './knn_model.pkl'
NN_MODEL_PATH = './nn_model.keras'
_nn_model = None


os.makedirs('debug_plots', exist_ok=True)

def load_and_process_mnist():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)
    return X_train, y_train, X_test, y_test


def load_neural_network_model():
    global _nn_model
    if _nn_model is not None:
        return _nn_model
        
    try:
        _nn_model = keras.models.load_model(NN_MODEL_PATH)
        print("Loaded neural network model")
        return _nn_model
    except:
        print("Error loading neural network model. Please ensure it has been trained and saved.")
        return None

def knn_model():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        X_train, y_train, _, _  = load_and_process_mnist()
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
    return model


def crop_digit(gray_image, binary):
    """
    Crops and centers a digit in a square frame with padding.

    """

    rows = np.any(binary, axis=1) # find bounding box of digit
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, gray_image.shape[0]-1)
    cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, gray_image.shape[1]-1)
    

    padding = int(min(gray_image.shape) * 0.08) # adds padding
    rmin = max(0, rmin - padding)
    rmax = min(gray_image.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(gray_image.shape[1] - 1, cmax + padding)
    

    height = rmax - rmin + 1 # selection square
    width = cmax - cmin + 1
    
    if height > width: # cropping
        diff = height - width
        cmin = max(0, cmin - diff // 2)
        cmax = min(gray_image.shape[1] - 1, cmax + (diff - diff // 2))
    elif width > height:
        diff = width - height
        rmin = max(0, rmin - diff // 2)
        rmax = min(gray_image.shape[0] - 1, rmax + (diff - diff // 2))
    

    cropped_image = gray_image[rmin:rmax+1, cmin:cmax+1]
    return cropped_image


def image_preprocessing(image, model_type='knn'):
    '''
    preprocesses the drawing into mnist-style format
    the image is first converted to grayscale, then thresholded
    and centered in a square frame with padding, then resized to 28x28
    and inverted, then gamma corrected and contrast fixed
    '''
    gray_image = image / 255.0 if image.max() > 1 else image.copy()
    
    threshold = 0.8 if gray_image.mean() > 0.5 else 0.2

    binary = gray_image < threshold if gray_image.mean() > 0.5 else gray_image > threshold

    centered_image = crop_digit(gray_image, binary) 

    processed_image = resize(centered_image, (28, 28))

    processed_image = util.invert(processed_image)
    
    gamma = 0.7
    processed_image = np.power(processed_image, gamma) # gamma correcting contrast
    p_min, p_max = np.percentile(processed_image, (5, 95)) # fix contrast

    processed_image = np.clip((processed_image - p_min) / (p_max - p_min), 0, 1)

    if model_type == 'nn':
        print("used NN")
        return processed_image.reshape(1, 784).astype(np.float32)

    processed_image = (processed_image * 255).astype(np.uint8)
    processed_image = processed_image.reshape(1,-1)
    
    return processed_image


def print_model_data(processed_image):
        print(f"shape {processed_image.shape}")
        print("Array data as 28x28 grid:")
        reshaped = processed_image.reshape(28, 28)
        print("     " + " ".join(f"{i:3d}" for i in range(28)))
        print("    " + "-" * 84)
        for i, row in enumerate(reshaped):
            print(f"{i:2d} | " + " ".join(f"{int(val):3d}" for val in row))


def predict_image(image, model_type='knn'):
    processed_image = image_preprocessing(image, model_type)

    if model_type == 'knn':
        
        model = knn_model()
        predicted_digit = model.predict(processed_image)
        print_model_data(processed_image)
        print(f"KNN prediction: {predicted_digit[0]}")

    elif model_type == 'nn':
        model = load_neural_network_model()
        predictions = model.predict(processed_image)
        print("Neural Network predictions:")
        for i, prob in enumerate(predictions[0]):
            print(f"  Digit {i}: {prob:.4f}")
        predicted_digit = np.argmax(predictions, axis=1)
    
    processed_image = image_preprocessing(image, "knn") # for plotting

    return predicted_digit, processed_image


def create_plot(left_image=None, right_image=None, nn_image=None, left_title="Input Drawing", knn_title="KNearest Prediction", nn_title="Neural Network Prediction", knn_prediction=None, nn_prediction=None):
    '''
    Plots the original image, KNN prediction, and NN prediction
    If empty images are passed, white images are created
    '''
    
    if knn_prediction is not None:
        knn_title = f"{knn_title}: {knn_prediction[0]}"
    
    if nn_prediction is not None:
        nn_title = f"{nn_title}: {nn_prediction[0]}"
    
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=(left_title, knn_title, nn_title),
                       column_widths=[0.33, 0.33, 0.33])
    
    white = np.ones((28, 28)) * 255 

    if left_image is None:
        left_image = white
    
    if right_image is None:
        right_image = white
        
    if nn_image is None:
        nn_image = white
    
    # Input image
    fig.add_trace(go.Heatmap(
        z=np.flipud(left_image), 
        colorscale='gray', 
        showscale=False,
        zmin=0,
        zmax=255
    ), row=1, col=1)
    
    # KNN prediction
    fig.add_trace(go.Heatmap(
        z=np.flipud(right_image), 
        colorscale='gray', 
        showscale=False,
        zmin=0,
        zmax=255
    ), row=1, col=2)
    
    # NN prediction
    fig.add_trace(go.Heatmap(
        z=np.flipud(nn_image), 
        colorscale='gray', 
        showscale=False,
        zmin=0,
        zmax=255
    ), row=1, col=3)


    fig.update_layout(
        height=400,
        width=1000,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    for col in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, row=1, col=col)
        

    fig.update_xaxes(constrain='domain', scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(constrain='domain', scaleanchor="x", scaleratio=1, row=1, col=1)
    
    fig.update_xaxes(constrain='domain', scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(constrain='domain', scaleanchor="x2", scaleratio=1, row=1, col=2)
    
    fig.update_xaxes(constrain='domain', scaleanchor="y3", scaleratio=1, row=1, col=3)
    fig.update_yaxes(constrain='domain', scaleanchor="x3", scaleratio=1, row=1, col=3)
    
    return fig


# def debug_plot(image, processed_image, predicted_digit):
#     plt.figure(figsize=(10, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title("Original Image")
#     plt.axis('off')
    
#     processed_image = processed_image.reshape(28, 28)
#     plt.subplot(1, 2, 2)
#     plt.imshow(processed_image, cmap='gray')
#     plt.title(f"Processed Image (Predicted: {predicted_digit[0]})")
#     plt.axis('off')
    
#     plt.tight_layout()
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"debug_plots/digit_{predicted_digit[0]}_{timestamp}.png"
#     plt.savefig(filename)
#     plt.close()
    
#     print(f"Debug plot saved to {filename}")
