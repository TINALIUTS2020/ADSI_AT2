from typing import Union

from api.DataModels import SingleInput, SingleResponse
from api.clean_string import clean_string
from api.tf_model import make_prediction

from fastapi import status, HTTPException

import pandas as pd


async def predict(data: Union[SingleInput, list], single_input=False):
    # pydantic should prevent malformed data
    # format strings
    # moved none repalcement and type setting to datamodel
    if isinstance(data, SingleInput):
        data = [data]

    parsed = [input.dict() for input in data]
    parsed = [{key: value if key != "brewery_name" else clean_string(value) for key, value in input.items()} for input in parsed]
    parsed = pd.DataFrame(parsed)

    predictions = await make_prediction(parsed)
    from logging import warning
    
    if predictions.size != parsed.shape[0]:
        raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="missmatched predictions", 
                )
        
    parsed["beer_style"] = predictions

    out = parsed.to_dict("records")
    out = [SingleResponse.parse_obj(compiled) for compiled in out]
    if single_input:
        out = out[0]
    return out 


