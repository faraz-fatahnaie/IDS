import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import (MinMaxScaler, LabelBinarizer)
from sklearn.model_selection import train_test_split
from utils import parse_data, save_dataframe, sort_columns


class BuildDataFrames:

    def __init__(self, dataframe=None, df_path: str = '', df_type: str = 'train', classification_mode: str = 'multi'):

        self.df_path = df_path
        self.df_type = df_type
        self.classification_mode = classification_mode
        self.label_feature_name = ' Label'
        self.DataFrame = None
        if dataframe is not None:
            self.DataFrame = dataframe

    def read_dataframe(self):

        cols_to_drop = [' Destination Port']
        df = {}
        total = 0
        for idx, file in enumerate(os.listdir(self.df_path)):
            df[idx] = pd.read_csv(Path(self.df_path).joinpath(file)).drop(cols_to_drop, axis=1)
            if idx == 0:
                self.DataFrame = pd.concat([self.DataFrame, df[idx]], axis=0)
            else:
                if df[idx].columns.all() == self.DataFrame.columns.all():
                    self.DataFrame = pd.concat([self.DataFrame, df[idx]], axis=0)

            df_size = len(df[idx])
            total += df_size
        # self.DataFrame = self.DataFrame.sample(n=100)
        if self.classification_mode == 'binary':
            self.DataFrame[self.label_feature_name] = self.DataFrame[self.label_feature_name].apply(
                lambda x: 1 if x != 'BENIGN' else 0)
        print('Total Number of Samples in Aggregated Dataframe is: ', total)
        self.DataFrame = self.DataFrame.replace([np.inf, -np.inf], 0)
        self.DataFrame = self.DataFrame.fillna(self.DataFrame.mean())
        train_file = os.path.join(Path(self.df_path).parent, 'cicids_describe.csv')
        self.DataFrame.describe().T.to_csv(train_file, index=True)
        return self.DataFrame

    def normalization(self, min_max_scaler_object=None):

        if self.df_type == 'train':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and self.label_feature_name in numeric_col_list:
                numeric_col_list = numeric_col_list.drop(self.label_feature_name)
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            min_max_scaler = MinMaxScaler()
            self.DataFrame[numeric_col_list] = min_max_scaler.fit_transform(DataFrame_numeric_col)
            return self.DataFrame, min_max_scaler

        elif self.df_type == 'test':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and self.label_feature_name in numeric_col_list:
                numeric_col_list = numeric_col_list.drop(self.label_feature_name)
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            self.DataFrame[numeric_col_list] = min_max_scaler_object.transform(DataFrame_numeric_col)
            return self.DataFrame

    def label_binarizing(self, label_binarizer=None):

        if self.df_type == 'train':
            # create an object of label binarizer, then fit on train labels
            LabelBinarizerObject_fittedOnTrainLabel = LabelBinarizer().fit(self.DataFrame[' Label'])
            # transform train labels with that object
            TrainBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(
                self.DataFrame[self.label_feature_name])
            # convert transformed labels to dataframe
            TrainBinarizedLabelDataFrame = pd.DataFrame(TrainBinarizedLabel,
                                                        columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)
            # concatenate training set after drop 'label' with created dataframe of binarized labels
            self.DataFrame = pd.concat([self.DataFrame.drop([self.label_feature_name], axis=1).reset_index(drop=True),
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
    dataset_path = base_path.joinpath('original')
    save_path = base_path.joinpath('preprocessed')
    classification_m = 'binary'
    # classification_m = 'multi'

    # =========== TRAIN DATAFRAME PREPROCESSING ===========
    preprocess_df = BuildDataFrames(df_path=str(dataset_path), df_type='train',
                                    classification_mode=classification_m)
    df = preprocess_df.read_dataframe()
    train, test = train_test_split(df, test_size=0.2)
    del df
    del preprocess_df

    preprocess_train = BuildDataFrames(dataframe=train, df_type='train', classification_mode=classification_m)
    train_tosave, min_max_scaler_obj = preprocess_train.normalization()

    # =========== TEST DATAFRAME PREPROCESSING ===========
    preprocess_test = BuildDataFrames(dataframe=test, df_type='test', classification_mode=classification_m)
    test_tosave = preprocess_test.normalization(min_max_scaler_object=min_max_scaler_obj)

    # =========== CHECKING ===========
    diff_test_train = list(set(test_tosave.columns) - set(train_tosave.columns))
    diff_train_test = list(set(train_tosave.columns) - set(test_tosave.columns))
    print('CHECKING => these should be two EMPTY lists:', diff_train_test, diff_test_train)

    # =========== LABEL BINARIZING FOR MULTI-CLASSIFICATION MODE ===========
    if classification_m == 'multi':
        train_tosave, label_binarizer_object = preprocess_train.label_binarizing()
        test_tosave = preprocess_test.label_binarizing(label_binarizer=label_binarizer_object)

    # =========== COLUMNS ORDER EQUALIZATION FOR FURTHER PICTURE FORMATTING ===========
    train_sorted, test_sorted = sort_columns(train_tosave, test_tosave)

    # =========== SAVE RESULTS ===========
    save_dataframe(train_tosave, save_path, 'train', classification_m)
    save_dataframe(test_tosave, save_path, 'test', classification_m)

    X, y = parse_data(train_tosave, dataset_name='CICIDS', classification_mode=classification_m)
    print(X.shape, y.shape)

    X, y = parse_data(test_tosave, dataset_name='CICIDS', classification_mode=classification_m)
    print(X.shape, y.shape)
