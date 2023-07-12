from typing import Union, List

from fastapi import FastAPI, status, Response, HTTPException

from starlette.responses import PlainTextResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

from api.DataModels import SingleInput, SingleResponse
from api.HomePage import HomePage
from api.Predictor import predict
from api.tf_model import get_architecture

from joblib import load
import pandas as pd


app = FastAPI(
    title="ADSI AT2 BEER API",
    description="This API provides predictions for beer types based on given input parameters.",
    summary="What beer are you drinking?",
    version="0.0.1",
    redoc_url=None,
)
# TODO: Add some api key in header request


# '/' (GET) - Display project information and endpoints
@app.get('/', include_in_schema=False)
async def home():
    # homepage = HomePage()
    # return HTMLResponse(content=homepage.page)
    return RedirectResponse(url='/docs')


# '/health/' (GET) - Return a welcome message
@app.get('/health/', status_code=status.HTTP_200_OK)
def health():
    return 'Welcome to the Beer API! The weights and biases have been brewed and are a-waiting your request'


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
    """
     Submit requests for predictions on single beer.

     Can use either query parameters or json body to submit request but not both.
    """
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
    
    if input_data:
        data = input_data

    elif any(request_vars):
        data = {
            "brewery_name": brewery_name,
            "review_aroma": review_aroma,
            "review_appearance": review_appearance,
            "review_palate": review_palate,
            "review_taste": review_taste,
            "beer_abv": beer_abv,
        }
        data = SingleInput.parse_obj(data)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="data could not be set, please raise issue on github with request example", 
        )

    # Perform prediction based on input data
    prediction = await predict(data, single_input=True)

    return prediction


# '/beers/type/' (POST) - Return predictions for multiple inputs
@app.post('/beers/type/', status_code=status.HTTP_200_OK)
async def predict_multiple(
    input_data: Union[List[SingleInput], None]=None,
) -> List[SingleResponse]:
    """
     Submit requests for predictions of multiple beers.

     Provide a list/array of where each element of the list is a of input variables to predict a beery type.
    """

    if input_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided", 
        )
    
    # Perform predictions based on input data
    # will be slow because handling per input
    prediction = await predict(input_data, single_input=False)
    return prediction


# '/model/architecture/' (GET) - Display the architecture of the Neural Networks
@app.get('/model/architecture/',
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
async def model_architecture():
    image_bytes: bytes = await get_architecture()
    return Response(content=image_bytes, media_type="image/png")