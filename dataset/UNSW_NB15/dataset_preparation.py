import pandas as pd

from sklearn.preprocessing import (MinMaxScaler, LabelBinarizer)
from pathlib import Path

from utils import parse_data, save_dataframe, sort_columns


class BuildDataFrames:

    def __init__(self, df_path: str, df_type: str = 'train', classification_mode: str = 'multi'):

        self.DataFrame = None
        self.df_path = df_path
        self.df_type = df_type
        self.classification_mode = classification_mode
        self.label_feature_name = 'attack_cat' if self.classification_mode == 'multi' else 'label'
        # self.label_feature_name = 'attack_cat'

    def read_dataframe(self):

        cols_to_drop = ['id', 'label'] if self.classification_mode == 'multi' else ['id', 'attack_cat']
        # cols_to_drop = ['id', 'label']
        self.DataFrame = pd.read_csv(self.df_path)
        self.DataFrame.drop(cols_to_drop, axis=1, inplace=True)
        # if classification_m == 'binary':
        #     self.DataFrame['attack_cat'] = self.DataFrame['attack_cat'].apply(lambda x: 'attack' if x != 'Normal' else x)
        return self.DataFrame

    def normalization(self, min_max_scaler_object=None):

        if self.df_type == 'train':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and 'label' in numeric_col_list:
                numeric_col_list = numeric_col_list.drop('label')
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            min_max_scaler = MinMaxScaler()
            DataFrame_numeric_col_scaled = min_max_scaler.fit_transform(DataFrame_numeric_col)
            DataFrame_numeric_col_scaled = pd.DataFrame(DataFrame_numeric_col_scaled, columns=numeric_col_list)
            self.DataFrame[numeric_col_list] = DataFrame_numeric_col_scaled
            return self.DataFrame, min_max_scaler
        elif self.df_type == 'test':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and 'label' in numeric_col_list:
                numeric_col_list = numeric_col_list.drop('label')
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            DataFrame_numeric_col_scaled = min_max_scaler_object.transform(DataFrame_numeric_col)
            DataFrame_numeric_col_scaled = pd.DataFrame(DataFrame_numeric_col_scaled, columns=numeric_col_list)
            self.DataFrame[numeric_col_list] = DataFrame_numeric_col_scaled
            return self.DataFrame

    def onehot_encoding(self):

        categorical_column_list = list(self.DataFrame.select_dtypes(include='object').columns)
        if self.classification_mode == 'multi':
            categorical_column_list.remove(self.label_feature_name)

        one_hot_DataFrame = pd.get_dummies(self.DataFrame, columns=categorical_column_list)
        one_hot_DataFrame_cols = one_hot_DataFrame.columns

        train_label_col = one_hot_DataFrame.pop(self.label_feature_name)

        # position parameter for insert function starts with 0 so,
        # we don't need to len(one_hot_train.columns)+1 (after pop function)
        one_hot_DataFrame.insert(len(one_hot_DataFrame.columns), self.label_feature_name, train_label_col)

        self.DataFrame = one_hot_DataFrame

        return self.DataFrame, one_hot_DataFrame_cols

    def column_equalization(self, train_one_hot_cols, test_one_hot_cols):

        if self.df_type == 'train':
            difference_test_train = list(set(test_one_hot_cols) - set(train_one_hot_cols))
            if len(difference_test_train) > 0:
                zero_values = [0] * len(self.DataFrame)
                for col in difference_test_train:
                    self.DataFrame[col] = zero_values

                # upcoming 2 lines of code, take attack_cat column and put it at the last column
                train_label_col = self.DataFrame.pop(self.label_feature_name)
                self.DataFrame.insert(len(self.DataFrame.columns), self.label_feature_name, train_label_col)
                return self.DataFrame
            else:
                return self.DataFrame

        elif self.df_type == 'test':
            difference_train_test = list(set(train_one_hot_cols) - set(test_one_hot_cols))
            if len(difference_train_test) > 0:
                zero_values = [0] * len(self.DataFrame)
                for col in difference_train_test:
                    self.DataFrame[col] = zero_values

                # upcoming 2 lines of code, take attack_cat column and put it at the last column
                test_label_col = self.DataFrame.pop(self.label_feature_name)
                self.DataFrame.insert(len(self.DataFrame.columns), self.label_feature_name, test_label_col)
                return self.DataFrame
            else:
                return self.DataFrame

    def label_binarizing(self, label_binarizer=None):

        if self.df_type == 'train':
            # create an object of label binarizer, then fit on train labels
            LabelBinarizerObject_fittedOnTrainLabel = LabelBinarizer().fit(self.DataFrame[self.label_feature_name])
            # transform train labels with that object
            TrainBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(
                self.DataFrame[self.label_feature_name])
            # convert transformed labels to dataframe
            TrainBinarizedLabelDataFrame = pd.DataFrame(TrainBinarizedLabel,
                                                        columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)
            # concatenate training set after drop 'label' with created dataframe of binarized labels
            self.DataFrame = pd.concat([self.DataFrame.drop([self.label_feature_name], axis=1),
                                        TrainBinarizedLabelDataFrame], axis=1)

            return self.DataFrame, LabelBinarizerObject_fittedOnTrainLabel

        elif self.df_type == 'test':

            TestBinarizedLabel = label_binarizer.transform(self.DataFrame[self.label_feature_name])
            TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel, columns=label_binarizer.classes_)
            self.DataFrame = pd.concat(
                [self.DataFrame.drop([self.label_feature_name], axis=1), TestBinarizedLabelDataFrame],
                axis=1)
            return self.DataFrame


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.joinpath('file')
    train_path = base_path.joinpath('original', 'UNSW_NB15_training-set.csv')
    test_path = base_path.joinpath('original', 'UNSW_NB15_testing-set.csv')
    save_path = base_path.joinpath('preprocessed')
    classification_m = 'binary'
    # classification_m = 'multi'

    # =========== TRAIN DATAFRAME PREPROCESSING ===========
    preprocess_train = BuildDataFrames(df_path=str(train_path), df_type='train',
                                       classification_mode=classification_m)
    train = preprocess_train.read_dataframe()
    train_normalized, min_max_scaler_obj = preprocess_train.normalization()
    train_one_hot, train_one_hot_cols = preprocess_train.onehot_encoding()

    # =========== TEST DATAFRAME PREPROCESSING ===========
    preprocess_test = BuildDataFrames(df_path=str(test_path), df_type='test',
                                      classification_mode=classification_m)
    test = preprocess_test.read_dataframe()
    test_normalized = preprocess_test.normalization(min_max_scaler_object=min_max_scaler_obj)
    test_one_hot, test_one_hot_cols = preprocess_test.onehot_encoding()

    # =========== FEATURE (COLUMN) EQUALIZATION BETWEEN TRAIN AND TEST DATAFRAMES  ===========
    train_rescaled = preprocess_train.column_equalization(train_one_hot_cols=train_one_hot_cols,
                                                          test_one_hot_cols=test_one_hot_cols)
    test_rescaled = preprocess_test.column_equalization(train_one_hot_cols=train_one_hot_cols,
                                                        test_one_hot_cols=test_one_hot_cols)

    # =========== CHECKING ===========
    diff_test_train = list(set(test_rescaled.columns) - set(train_rescaled.columns))
    diff_train_test = list(set(train_rescaled.columns) - set(test_rescaled.columns))
    print('CHECKING => these should be two EMPTY lists:', diff_train_test, diff_test_train)

    train_temp, test_temp = train_rescaled, test_rescaled

    # =========== LABEL BINARIZE FOR MULTI-CLASSIFICATION MODE ===========
    if classification_m == 'multi':
        train_label_binarized, label_binarizer_object = preprocess_train.label_binarizing()
        test_label_binarized = preprocess_test.label_binarizing(label_binarizer=label_binarizer_object)
        train_temp, test_temp = train_label_binarized, test_label_binarized

    # =========== COLUMNS ORDER EQUALIZATION FOR FURTHER PICTURE FORMATTING ===========
    train_sorted, test_sorted = sort_columns(train_temp, test_temp)

    # =========== SAVE RESULTS ===========
    save_dataframe(train_sorted, save_path, 'train', classification_m)
    save_dataframe(test_sorted, save_path, 'test', classification_m)

    X, y = parse_data(train_sorted, dataset_name='UNSW_NB15', classification_mode=classification_m)
    print(f'train shape: x=>{X.shape}, y=>{y.shape}')

    X, y = parse_data(test_sorted, dataset_name='UNSW_NB15', classification_mode=classification_m)
    print(f'test shape: x=>{X.shape}, y=>{y.shape}')
