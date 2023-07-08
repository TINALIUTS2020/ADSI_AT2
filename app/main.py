from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

#model_name = load('../model_path.joblib')

# '/' (GET) - Display project information and endpoints
@app.get('/')
def home():
    info = {
        'description': 'This API provides predictions for beer types based on given input parameters.',
        'endpoints': {
            '/': 'Display project information and endpoints',
            '/health/': 'Return a welcome message',
            '/beer/type/': 'Return prediction for a single input',
            '/beers/type/': 'Return predictions for multiple inputs',
            '/model/architecture/': 'Display the architecture of the Neural Networks'
        },
        'input_parameters': {
            'brewery_name': 'string',
            'review_aroma': 'float',
            'review_appearance': 'float',
            'review_palate': 'float',
            'review_taste': 'float',
            'beer_abv': 'float'
        },
        'output_format': 'JSON',
        'github_repo': 'https://github.com/your-github-repo'
    }
    return info


# '/health/' (GET) - Return a welcome message
@app.get('/health/')
def health():
    return 'Welcome to the Beer API!'


# '/beer/type/' (POST) - Return prediction for a single input
@app.post('/beer/type/')
def predict_single():
    data = request.json()
    # Perform prediction based on input data
    prediction = {
        'beer_type': 'predicted_type'
    }
    return prediction


# '/beers/type/' (POST) - Return predictions for multiple inputs
@app.post('/beers/type/')
def predict_multiple():
    data = request.json()
    # Perform predictions based on input data
    predictions = [
        {'beer_type': 'predicted_type_1'},
        {'beer_type': 'predicted_type_2'},
        {'beer_type': 'predicted_type_3'}
    ]
    return predictions


# '/model/architecture/' (GET) - Display the architecture of the Neural Networks
@app.get('/model/architecture/')
def model_architecture():
    architecture = {
        'layers': [
            {'name': 'layer_1', 'type': 'type_1'},
            {'name': 'layer_2', 'type': 'type_2'},
            {'name': 'layer_3', 'type': 'type_3'}
        ]
    }
    return architecture
