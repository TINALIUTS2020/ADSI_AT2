from api.DataModels import SingleInput, SingleResponse

async def predict(data: SingleInput):
    data = data.dict()
    data["beer_type"] = "test_type"
    return SingleResponse.parse_obj(data)