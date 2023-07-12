from typing import Union

from api.DataModels import SingleInput, SingleResponse
from api.clean_string import clean_string
from api.tf_model import make_prediction

from fastapi import status, HTTPException

async def predict(data: Union[SingleInput, list]):
    # pydantic should prevent malformed data
    # format strings
    # moved none repalcement and type setting to datamodel
    ref = {field: [] for field in SingleInput.__fields__.keys()}

    if isinstance(data, SingleInput):
        data = [data]

    parsed = [input.dict() for input in data]

    for input in parsed:
        for key, value in input.items():
            if key == "brewery_name":
                value = clean_string(value)
            ref[key].append(value)

    predictions = await make_prediction(ref)
    
    if len(predictions) != len(parsed):
        raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="missmatched predictions", 
                )

    for input_data, pred_out in zip(parsed, predictions):
        input_data["beer_style"] = pred_out

    out = [SingleResponse.parse_obj(compiled) for compiled in parsed]
    return out 


