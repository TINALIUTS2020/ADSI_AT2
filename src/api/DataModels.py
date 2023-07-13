from typing import Union

from pydantic import BaseModel, Field, root_validator, validator
from pydantic.dataclasses import dataclass


# @dataclass
class SingleInput(BaseModel):
    # TODO: get accetable ranges for floats
    brewery_name: Union[str, None] = Field(default=None, description="Name of brewery", max_length=100)
    review_aroma: Union[float, None] = Field(default=None, description="How good does the beer smell", ge=1, le=5)
    review_appearance: Union[float, None] = Field(default=None, description="How good does the beer look", ge=1, le=5)
    review_palate: Union[float, None] = Field(default=None, description="Something else to do with taste?", ge=1, le=5)
    review_taste: Union[float, None] = Field(default=None, description="How good does the beer taste?", ge=1, le=5)
    beer_abv: Union[float, None] = Field(default=None, description="Alchohol by volume", ge=0, le=100)

    @root_validator(skip_on_failure=True, pre=False)
    def any_of(cls, v):
        if not any(v.values()):
            raise ValueError('atleast one field must be filled')

        for key, value in v.items():
            if value is None:
                if key == "brewery_name":
                    v[key] = "[UNK]"
                else:
                    v[key] = float(-9)

        return v

    # TODO: update example
    class Config:
        validate_assignment=False
        schema_extra = {
            "examples": [
                {
                    "brewery_name": "Amazing But Not Pretty Example Brews", # name
                    "review_aroma": 5,
                    "review_appearance": 1,
                    "review_palate": 4,
                    "review_taste": 4,
                    "beer_abv": 2.4,
                }
            ]
        }

                # {
                #     "brewery_name": "Pretty But Not Amazing Mid Strength Example Brews", # name
                #     "review_aroma": 1,
                #     "review_appearance": 5,
                #     "review_palate": 2,
                #     "review_taste": 1,
                #     "beer_abv": 0.5,
                #     "beer_type": "Best IPA",
                # },

class SingleResponse(BaseModel):
    brewery_name: Union[str, None] = Field(default=None, description="Name of brewery", max_length=100)
    review_aroma: Union[float, None] = Field(default=None, description="How good does the beer smell", ge=1, le=5)
    review_appearance: Union[float, None] = Field(default=None, description="How good does the beer look", ge=1, le=5)
    review_palate: Union[float, None] = Field(default=None, description="Something else to do with taste?", ge=1, le=5)
    review_taste: Union[float, None] = Field(default=None, description="How good does the beer taste?", ge=1, le=5)
    beer_abv: Union[float, None] = Field(default=None, description="Alchohol by volume", ge=0, le=100)
    beer_style: str = Field(description="Predicted style of beer")

    @root_validator(skip_on_failure=True, pre=True)
    def any_of(cls, v):
        for key, value in v.items():
                if key == "brewery_name" and value == "[UNK]":
                    v[key] = None
                elif value == -9:
                    v[key] = None

        return v

    # TODO: update example
    class Config:
        schema_extra = {
            "examples": [
                {
                    "brewery_name": "Amazing But Not Pretty Example Brews", # name
                    "review_aroma": 5,
                    "review_appearance": 1,
                    "review_palate": 4,
                    "review_taste": 4,
                    "beer_abv": 2.4,
                    "beer_style": "Best IPA",
                },
            ]
        }

class MultipleInput(BaseModel):
    pass

class MultipleResponse(MultipleInput):
    pass