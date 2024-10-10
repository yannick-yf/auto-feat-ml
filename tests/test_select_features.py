from unittest import TestCase
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from auto_feat_ml import FeatureSelection
from auto_feat_ml.data_models.feature_model import FeatureIn, FeatureOut
from sklearn.model_selection import GroupKFold

class TestFeatureSelection(TestCase):
    def setUp(self) -> None:
        self.n_samples = 500
        self.n_features=20
        self.noise=0.1
        self.n_informative=5
        self.random_state=42

    def _get_training_set(self, selection_type: str)->pd.DataFrame:
        # generate regression dataset
        if selection_type =='regression':
            X, y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                noise=self.noise,
                n_informative=self.n_informative,
                random_state=self.random_state,
            )
        elif selection_type =='classification':
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=self.n_informative,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Only 'classification', 'regression' selection_type is supported")

        X = pd.DataFrame(X)

        prefix = 'column_'

        # Using List Comprehension
        X.columns = [prefix + str(i) for i in X.columns]

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1
        )

        # Transform numpy.ndarray to pd.DataFrame
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)

        return X_train, y_train

    def test_regression_select_features_pipeline_w_group_kfold(self):
        """
        GIVEN a FeatureSelection object with a user group
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10 and the kfold method is used
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)
        list_nb_feature_to_select = [5]
        # step_rfe = 0.1

        X_train["trial_id"] = 'categ_0'
        X_train["trial_id"] = np.where(
            X_train["column_1"] < 0.5, 
            'categ_1', np.where(X_train["column_1"] > 0.8, 'categ_3', X_train["trial_id"]))

        groups = X_train["trial_id"]
        X_train = X_train.drop(["trial_id"], axis=1)
        cv = GroupKFold(n_splits=3)

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                selection_type = selection_type
                )
        )

        output = feature_selection.select_features_pipeline(
            pd_column_groups=groups,
            group_kfold=cv
            )

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_regression_select_features_pipeline(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [15]

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_regression_select_features_pipeline_w_force_features(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [3]

        col_to_force =  ['column_1', 'column_18']

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                features_to_force = col_to_force,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        condition_max_features = int(max(list_nb_feature_to_select) + len(col_to_force))

        assert len(output.column_names) <=  condition_max_features and set(col_to_force).issubset(output.column_names) is True

    def test_regression_select_features_pipeline_w_wrong_force_features(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [3]

        col_to_force =  ['data_science']

        with self.assertRaises(ValueError):
            FeatureSelection(
                    FeatureIn(
                        list_number_feature_to_select=list_nb_feature_to_select,
                        training_set=X_train,
                        target_variable=y_train,
                        features_to_force = col_to_force,
                        selection_type=selection_type
                        )
                ).select_features_pipeline()

    def test_regression_select_features_pipeline_int_step_rfe(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called with step_rfe int is provided
        THEN check that the returned list of column names is of length 10
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)
        step_rfe = 2
        list_nb_feature_to_select = [15]

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                step_rfe=step_rfe,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_regression_select_features_pipeline_float_step_rfe(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called with step_rfe float is provided
        THEN check that the returned list of column names is of length 10
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [15]
        step_rfe = 0.1

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                step_rfe=step_rfe,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_regression_training_set_dtype_validation(self):
        """
        GIVEN a FeatureSelection object using a dataframe with an object column
        WHEN the select_features_pipeline method is called
        THEN the method return ValueError
        """
        selection_type='regression'
        X_train, y_train = self._get_training_set(selection_type)
        
        list_nb_feature_to_select = [15]

        X_train['column_1'] = np.where(
            X_train['column_1']<0.5,
            'categ1',
            'categ2')
        
        with self.assertRaises(ValueError):
            FeatureSelection(
                    FeatureIn(
                        list_number_feature_to_select=list_nb_feature_to_select,
                        training_set=X_train,
                        target_variable=y_train
                        )
                ).select_features_pipeline()
            
    def test_classification_select_features_pipeline_w_group_kfold(self):
        """
        GIVEN a FeatureSelection object with a user group
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10 and the kfold method is used
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)
        list_nb_feature_to_select = [5]
        # step_rfe = 0.1

        X_train["trial_id"] = 'categ_0'
        X_train["trial_id"] = np.where(
            X_train["column_1"] < 0.5, 
            'categ_1', np.where(X_train["column_1"] > 0.8, 'categ_3', X_train["trial_id"]))

        groups = X_train["trial_id"]
        X_train = X_train.drop(["trial_id"], axis=1)
        cv = GroupKFold(n_splits=3)

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                selection_type = selection_type
                )
        )

        output = feature_selection.select_features_pipeline(
            pd_column_groups=groups,
            group_kfold=cv
            )

        assert len(output.column_names) <= max(list_nb_feature_to_select) 

    def test_select_features_pipeline(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [15]

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_select_features_pipeline_w_force_features(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [3]

        col_to_force =  ['column_1', 'column_18']

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                training_set=X_train,
                target_variable=y_train,
                features_to_force = col_to_force,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        condition_max_features = int(max(list_nb_feature_to_select) + len(col_to_force))

        assert len(output.column_names) <=  condition_max_features and set(col_to_force).issubset(output.column_names) is True

    def test_select_features_pipeline_w_wrong_force_features(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called
        THEN check that the returned list of column names is of length 10
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [3]

        col_to_force =  ['data_science']

        with self.assertRaises(ValueError):
            FeatureSelection(
                    FeatureIn(
                        list_number_feature_to_select=list_nb_feature_to_select,
                        training_set=X_train,
                        target_variable=y_train,
                        features_to_force = col_to_force,
                        selection_type=selection_type
                        )
                ).select_features_pipeline()

    def test_select_features_pipeline_int_step_rfe(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called with step_rfe int is provided
        THEN check that the returned list of column names is of length 10
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)
        step_rfe = 2
        list_nb_feature_to_select = [15]

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                step_rfe=step_rfe,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_select_features_pipeline_float_step_rfe(self):
        """
        GIVEN a FeatureSelection object
        WHEN the select_features_pipeline method is called with step_rfe float is provided
        THEN check that the returned list of column names is of length 10
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)

        list_nb_feature_to_select = [15]
        step_rfe = 0.1

        feature_selection = FeatureSelection(
            FeatureIn(
                list_number_feature_to_select=list_nb_feature_to_select,
                step_rfe=step_rfe,
                training_set=X_train,
                target_variable=y_train,
                selection_type=selection_type
                )
        )

        output = feature_selection.select_features_pipeline()

        assert len(output.column_names) <= max(list_nb_feature_to_select)

    def test_training_set_dtype_validation(self):
        """
        GIVEN a FeatureSelection object using a dataframe with an object column
        WHEN the select_features_pipeline method is called
        THEN the method return ValueError
        """
        selection_type='classification'
        X_train, y_train = self._get_training_set(selection_type)
        
        list_nb_feature_to_select = [15]

        X_train['column_1'] = np.where(
            X_train['column_1']<0.5,
            'categ1',
            'categ2')
        
        with self.assertRaises(ValueError):
            FeatureSelection(
                    FeatureIn(
                        list_number_feature_to_select=list_nb_feature_to_select,
                        training_set=X_train,
                        target_variable=y_train,
                        selection_type=selection_type
                        )
                ).select_features_pipeline()
