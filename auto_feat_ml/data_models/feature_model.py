import numpy
import pandas as pd
from typing import List, Union
from pydantic import BaseModel, ConfigDict
from enum import Enum


class SelectionTypeEnum(str, Enum):
    regression = "regression"
    classification = "classification"


class FeatureIn(BaseModel):
    list_number_feature_to_select: List[int] = [10]
    step_rfe: Union[int, float] = 1
    training_set: pd.DataFrame
    target_variable: Union[numpy.ndarray, pd.Series]
    features_to_force: List[str] = None
    selection_type: str = SelectionTypeEnum.regression

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FeatureOut(BaseModel):
    column_names: list
    best_params: object
    best_score: float
