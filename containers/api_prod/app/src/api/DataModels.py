from typing import Union

from pydantic import BaseModel

class SingleInput(BaseModel):
    # TODO: get accetable ranges for floats
    brewery_name: Union[str, None] = None
    review_aroma: Union[float, None] = None
    review_appearance: Union[float, None] = None
    review_palate: Union[float, None] = None
    review_taste: Union[float, None] = None
    beer_abv: Union[float, None] = None

    # TODO: update example
    class Config:
        schema_extra = {
            "examples": [
                {
                    "brewery_name": "Foo",
                    "review_aroma": "A very nice Item",
                    "review_appearance": 3.6,
                    "review_palate": 3.2,
                    "review_taste": 1.5,
                    "beer_abv": 2.4,
                }
            ]
        }

class SingleResponse(SingleInput):
    beer_type: str

    # TODO: update example
    class Config:
        schema_extra = {
            "examples": [
                {
                    "brewery_name": "Foo",
                    "review_aroma": "A very nice Item",
                    "review_appearance": 3.6,
                    "review_palate": 3.2,
                    "review_taste": 1.5,
                    "beer_abv": 2.4,
                    "beer_type": "Best IPA",
                }
            ]
        }

class MultipleInput(BaseModel):
    pass

class MultipleResponse(MultipleInput):
    pass