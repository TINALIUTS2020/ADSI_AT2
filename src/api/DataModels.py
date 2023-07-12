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
    beer_abv: Union[float, None] = Field(default=None, description="Alchohol by volume", ge=1, le=5)

    @root_validator
    def any_of(cls, v):
        if not any(v.values()):
            raise ValueError('atleast one field must be filled')
        return v
    
    @validator("brewery_name", always=True)
    def set_brewery_name(cls, v):
        return v or "[UNK]"
    
    @validator("review_aroma", always=True)
    def set_review_aroma(cls, v):
        return v or float(-9)
    
    @validator("review_appearance", always=True)
    def set_review_appearance(cls, v):
        return v or float(-9)
    
    @validator("review_palate", always=True)
    def set_review_palate(cls, v):
        return v or float(-9)
    
    @validator("review_taste", always=True)
    def set_review_taste(cls, v):
        return v or float(-9)
    
    @validator("beer_abv", always=True)
    def set_beer_abv(cls, v):
        return v or float(-9)




    # TODO: update example
    class Config:
        validate_assignment = True
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

class SingleResponse(SingleInput):
    beer_type: str

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
                    "beer_type": "Best IPA",
                },
            ]
        }

class MultipleInput(BaseModel):
    pass

class MultipleResponse(MultipleInput):
    pass