from typing import Union, List

from fastapi import FastAPI, status, Response, HTTPException

from starlette.responses import PlainTextResponse
from pydantic import BaseModel

from api.DataModels import SingleInput, SingleResponse

from joblib import load
import pandas as pd


app = FastAPI()
# TODO: Add some api key in header request

#model_name = load('MODEL PATH.joblib')

# '/' (GET) - Display project information and endpoints
@app.get('/')
async def home():
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
        'github_repo': 'https://github.com/TINALIUTS2020/ADSI_AT2'
    }
    return info


# '/health/' (GET) - Return a welcome message
@app.get('/health/', status_code=status.HTTP_200_OK)
def health():
    return 'Welcome to the Beer API! The Nueral is brewed and awaiting your request'


# '/beer/type/' (POST) - Return prediction for a single input
@app.post('/beer/type/', status_code=status.HTTP_200_OK)
async def predict_single(
    input_data: Union[SingleInput, None]=None,
    brewery_name: Union[str, None]=None,
    review_aroma: Union[float, None]=None,
    review_appearance: Union[float, None]=None,
    review_palate: Union[float, None]=None,
    review_taste: Union[float, None]=None,
    beer_abv: Union[float, None]=None,
) -> SingleResponse:
    
    request_vars = [
        brewery_name,
        review_aroma,
        review_appearance,
        review_palate,
        review_palate,
        review_taste,
        beer_abv,
    ]
    request_options = request_vars + [input_data]
    
    if all(arg is None for arg in request_options):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided", 
        )
    
    if input_data and any(request_vars):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provide either data in body json or as query parameters not both", 
        )
    
    #TODO: error handle for if all of the json body is none
    
    if input_data:
        data = input_data

    if any(request_vars):
        data = {
            "brewery_name": brewery_name,
            "review_aroma": review_aroma,
            "review_appearance": review_appearance,
            "review_palate": review_palate,
            "review_taste": review_taste,
            "beer_abv": beer_abv,
        }
        data = SingleInput.model_validate(data)

    prediction = await predict(data)
    # Perform prediction based on input data

    return prediction


# '/beers/type/' (POST) - Return predictions for multiple inputs
@app.post('/beers/type/', status_code=status.HTTP_200_OK)
def predict_multiple(
    input_data: Union[List[SingleInput], None]=None,
):

    if all(arg is None for arg in locals().values()):

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