"""Feature Selection Class

For a given X_train, Y_train return a specific number of columms from a feature selection process
"""


import numpy as np
import pandas as pd
from typing import List, Union

from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    f_classif,
    mutual_info_classif,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
from mlxtend.feature_selection import ColumnSelector

from feature_engine.selection import DropDuplicateFeatures, DropConstantFeatures

from auto_feat_ml.data_models.feature_model import FeatureIn, FeatureOut

class FeatureSelection:
    def __init__(self, feature_object: FeatureIn) -> None:
        self.feature_object = feature_object

    def select_features_pipeline(
        self,
        pd_column_groups: pd.Series | None = None,
        group_kfold: GroupKFold | None = None,
    ) -> FeatureOut:
        self._get_training_set_dtype_validation()
        self._get_features_to_force_value_validation()

        pipeline_config = self._get_pipeline_config()

        pipe = pipeline_config["pipeline_def"]()
        search_space = pipeline_config["search_space"]()

        clf = self._grid_search_cv_fit(
            pipe, 
            search_space, 
            group_kfold, 
            scoring=pipeline_config["scoring"], 
            pd_column_groups=pd_column_groups)
        
        list_column_names = self._get_list_feature_selected(clf)

        return FeatureOut(
            column_names=list_column_names,
            best_params=clf.best_params_,
            best_score=clf.best_score_,
        )
    
    def _get_list_feature_selected(self, estimator):
        if self.feature_object.features_to_force is None:
            trainingset_fs = self.feature_object.training_set.iloc[
                :, estimator.best_estimator_.named_steps["selector"].get_support(indices=True)
            ]

            list_column_names = list(trainingset_fs.columns)

        else:
            trainingset_fs = self.feature_object.training_set.iloc[
                :,
                estimator.best_estimator_.named_steps["feats"]["feature_selection_pipeline"][
                    "selector"
                ].get_support(indices=True),
            ]

            list_column_names = list(
                set(
                    list(trainingset_fs.columns) + self.feature_object.features_to_force
                )
            )

        return list_column_names


    def _get_pipeline_config(self):
        config = {
            "regression": {
                "pipeline_def": self._get_regression_feature_selection_pipeline_definition,
                "search_space": self._get_regression_feature_selection_pipeline_search_space,
                "scoring": "r2",
            },
            "classification": {
                "pipeline_def": self._get_classification_feature_selection_pipeline_definition,
                "search_space": self._get_classification_feature_selection_pipeline_search_space,
                "scoring": "accuracy",
            },
        }

        if self.feature_object.selection_type not in config:
            raise ValueError(
                "Only 'classification', 'regression' selection_type is supported"
            )

        return config[self.feature_object.selection_type]

    def _grid_search_cv_fit(self,
        pipe: Pipeline,
        search_space: list,
        group_kfold: GroupKFold,
        scoring: str,
        pd_column_groups: Union[np.ndarray, pd.Series],
    ) -> GridSearchCV:
        clf = GridSearchCV(
            pipe,
            search_space,
            cv=group_kfold if group_kfold is not None else 5,
            verbose=3,
            scoring=scoring,
            n_jobs=-1,
        ).fit(
            self.feature_object.training_set,
            self.feature_object.target_variable,
            groups=pd_column_groups,
        )

        return clf

    def _get_training_set_dtype_validation(self):
        if isinstance(self.feature_object.training_set, pd.DataFrame):
            my_type = ["float64", "int64"]
            dtypes = self.feature_object.training_set.dtypes.to_dict()

            for col_name, typ in dtypes.items():
                if typ not in my_type:
                    raise ValueError(
                        f"Column name - `dataframe['{col_name}'].dtype == {typ}` not {my_type}"
                    )
        else:
            raise ValueError("Training Dataset needs to be a pd.DataFrame")

    def _get_features_to_force_value_validation(self):
        if self.feature_object.features_to_force is not None:
            if (
                set(self.feature_object.features_to_force).issubset(
                    self.feature_object.training_set.columns
                )
                is False
            ):
                raise ValueError(
                    "Feature force to be returned by the users is not in the Training Dataset provided"
                )

    def _get_regression_feature_selection_pipeline_definition(self) -> Pipeline:
        common_steps = [
            ("constant", DropConstantFeatures(tol=0.90)),
            ("duplicated", DropDuplicateFeatures()),
            (
                "selector",
                SelectKBest(
                    f_regression,
                    k=self.feature_object.list_number_feature_to_select[0],
                ),
            ),
        ]

        if self.feature_object.features_to_force is None:
            pipeline_steps = common_steps + [
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(max_depth=3, n_estimators=500)),
            ]
        else:
            feature_selection_pipe = Pipeline(common_steps)
            user_selection_pipe = Pipeline(
                [
                    (
                        "col_select_user",
                        ColumnSelector(cols=self.feature_object.features_to_force),
                    ),
                ]
            )

            pipeline_steps = [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("feature_selection_pipeline", feature_selection_pipe),
                            ("col_select_user", user_selection_pipe),
                        ]
                    ),
                ),
                ("scaler", StandardScaler()),
                ("duplicated", DropDuplicateFeatures()),
                ("regressor", RandomForestRegressor(max_depth=3, n_estimators=500)),
            ]

        return Pipeline(pipeline_steps)

    def _get_classification_feature_selection_pipeline_definition(self) -> Pipeline:
        common_steps = [
            ("constant", DropConstantFeatures(tol=0.90)),
            ("duplicated", DropDuplicateFeatures()),
            (
                "selector",
                SelectKBest(
                    f_classif,
                    k=self.feature_object.list_number_feature_to_select[0],
                ),
            ),
        ]

        if self.feature_object.features_to_force is None:
            pipeline_steps = common_steps + [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(max_depth=3, n_estimators=500)),
            ]
        else:
            feature_selection_pipe = Pipeline(common_steps)
            user_selection_pipe = Pipeline(
                [
                    (
                        "col_select_user",
                        ColumnSelector(cols=self.feature_object.features_to_force),
                    ),
                ]
            )

            pipeline_steps = [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("feature_selection_pipeline", feature_selection_pipe),
                            ("col_select_user", user_selection_pipe),
                        ]
                    ),
                ),
                ("scaler", StandardScaler()),
                ("duplicated", DropDuplicateFeatures()),
                ("classifier", RandomForestClassifier(max_depth=3, n_estimators=500)),
            ]

        return Pipeline(pipeline_steps)

    def _get_regression_feature_selection_pipeline_search_space(self) -> list:
        if self.feature_object.features_to_force is None:
            return [
                {
                    "selector": [
                        SelectKBest(f_regression),
                        SelectKBest(mutual_info_regression),
                    ],
                    "selector__k": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "constant": [None, DropConstantFeatures(tol=0.90)],
                },
                {
                    "selector": [
                        RFE(
                            estimator=RandomForestRegressor(max_depth=3, n_estimators=500),
                            step=self.feature_object.step_rfe,
                        )
                    ],
                    "selector__n_features_to_select": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "constant": [None, DropConstantFeatures(tol=0.90)],
                },
            ]
        else:
            return [
                {
                    "feats__feature_selection_pipeline__selector": [
                        SelectKBest(f_regression),
                        SelectKBest(mutual_info_regression),
                    ],
                    "feats__feature_selection_pipeline__selector__k": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "feats__feature_selection_pipeline__constant": [
                        None,
                        DropConstantFeatures(tol=0.90),
                    ],
                },
                {
                    "feats__feature_selection_pipeline__selector": [
                        RFE(estimator=RandomForestRegressor(max_depth=3, n_estimators=500))
                    ],
                    "feats__feature_selection_pipeline__selector__n_features_to_select": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "feats__feature_selection_pipeline__constant": [
                        None,
                        DropConstantFeatures(tol=0.90),
                    ],
                },
            ]

    def _get_classification_feature_selection_pipeline_search_space(self) -> list:
        if self.feature_object.features_to_force is None:
            return [
                {
                    "selector": [
                        SelectKBest(f_classif),
                        SelectKBest(mutual_info_classif),
                    ],
                    "selector__k": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "constant": [None, DropConstantFeatures(tol=0.90)],
                },
                {
                    "selector": [
                        RFE(
                            estimator=RandomForestClassifier(max_depth=3, n_estimators=500),
                            step=self.feature_object.step_rfe,
                        )
                    ],
                    "selector__n_features_to_select": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "constant": [None, DropConstantFeatures(tol=0.90)],
                },
            ]
        else:
            return [
                {
                    "feats__feature_selection_pipeline__selector": [
                        SelectKBest(f_classif),
                        SelectKBest(mutual_info_classif),
                    ],
                    "feats__feature_selection_pipeline__selector__k": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "feats__feature_selection_pipeline__constant": [
                        None,
                        DropConstantFeatures(tol=0.90),
                    ],
                },
                {
                    "feats__feature_selection_pipeline__selector": [
                        RFE(estimator=RandomForestClassifier(max_depth=3, n_estimators=500))
                    ],
                    "feats__feature_selection_pipeline__selector__n_features_to_select": self.feature_object.list_number_feature_to_select,
                    "scaler": [None, StandardScaler()],
                    "feats__feature_selection_pipeline__constant": [
                        None,
                        DropConstantFeatures(tol=0.90),
                    ],
                },
            ]
