import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import (MinMaxScaler, LabelBinarizer)
from utils import parse_data


class BuildDataFrames:

    def __init__(self, dataframe=None, df_path: str = '', df_type: str = 'train', classification_mode: str = 'multi'):

        self.df_path = df_path
        self.df_type = df_type
        self.classification_mode = classification_mode
        self.label_feature_name = ' Label'
        self.DataFrame = None
        if dataframe is not None:
            self.DataFrame = dataframe

    def ReadDataFrames(self):

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
        # self.DataFrame = self.DataFrame.sample(n=1000000, random_state=np.random.seed(1400))
        if self.classification_mode == 'binary':
            self.DataFrame[self.label_feature_name] = self.DataFrame[self.label_feature_name].apply(
                lambda x: 1 if x != 'BENIGN' else 0)
        print('Total Number of Samples in Aggregated Dataframe is: ', total)
        self.DataFrame = self.DataFrame.replace([np.inf, -np.inf], 0)
        self.DataFrame = self.DataFrame.fillna(0)
        train_file = os.path.join('/home/faraz/PycharmProjects/IDS/dataset/CICIDS/file', 'cicids_describe.csv')
        self.DataFrame.describe().T.to_csv(train_file, index=True)
        return self.DataFrame

    def Normalization(self, min_max_scaler_object=None):

        if self.df_type == 'train':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and self.label_feature_name in numeric_col_list:
                numeric_col_list = numeric_col_list.drop(self.label_feature_name)
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            min_max_scaler = MinMaxScaler()
            self.DataFrame[numeric_col_list] = min_max_scaler.fit_transform(DataFrame_numeric_col)
            # DataFrame_numeric_col_scaled = pd.DataFrame(DataFrame_numeric_col_scaled, columns=numeric_col_list)
            # labels = pd.DataFrame(self.DataFrame[' Label'].values)
            # print(DataFrame_numeric_col_scaled)
            # print(len(self.DataFrame[' Label']))
            # self.DataFrame[numeric_col_list] = DataFrame_numeric_col_scaled

            # print(len(DataFrame_numeric_col_scaled), len(self.DataFrame[' Label']))
            # print(DataFrame_numeric_col_scaled.head())
            # print(self.DataFrame[' Label'].head())
            # self.DataFrame = pd.concat([DataFrame_numeric_col_scaled, labels], axis=1)

            return self.DataFrame, min_max_scaler
        elif self.df_type == 'test':
            DataFrame = self.DataFrame
            numeric_col_list = DataFrame.select_dtypes(include='number').columns
            if self.classification_mode == 'binary' and self.label_feature_name in numeric_col_list:
                numeric_col_list = numeric_col_list.drop(self.label_feature_name)
            DataFrame_numeric_col = DataFrame[numeric_col_list].values
            DataFrame_numeric_col_scaled = min_max_scaler_object.transform(DataFrame_numeric_col)
            DataFrame_numeric_col_scaled = pd.DataFrame(DataFrame_numeric_col_scaled, columns=numeric_col_list)
            self.DataFrame[numeric_col_list] = DataFrame_numeric_col_scaled
            return self.DataFrame

    def LabelBinarize(self, LabelBinarizerObj=None):

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

            TestBinarizedLabel = LabelBinarizerObj.transform(self.DataFrame[self.label_feature_name])
            TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel, columns=LabelBinarizerObj.classes_)
            self.DataFrame = pd.concat(
                [self.DataFrame.drop([self.label_feature_name], axis=1), TestBinarizedLabelDataFrame],
                axis=1)
            return self.DataFrame


def PictureFormat(DataFrame, classification_mode: str):  # do PictureFormat after LabelBinarize and Sorting
    # this dataset contains 10 different labels
    class_columns = DataFrame.columns[-10:] if classification_mode == 'multi' else DataFrame.columns[-1]
    X = DataFrame.drop(class_columns, axis=1)
    y = DataFrame[class_columns]

    X = np.array(X)
    y = np.array(y)
    print('Data has shape of:', np.shape(X), '\nLabel has shape of:', np.shape(y))

    X = np.reshape(X, (X.shape[0], 14, 14, 1))
    print('X shape is:', np.shape(X), '\ny shape is:', np.shape(y))

    return X, y


def SaveDataFrames(DataFrame, DataFrameType, classification_mode, save_path):
    file_name = DataFrameType
    if classification_mode == 'binary':
        file_name = file_name + '_binary'
    elif classification_mode == 'multi':
        file_name = file_name + '_multi'
    train_file = os.path.join(save_path, file_name + '.csv')
    DataFrame.to_csv(train_file, index=False)

    print('Saved:', train_file)


from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    base_dir = '/home/faraz/PycharmProjects/IDS/dataset/CICIDS/file/original/'
    save_path = '/home/faraz/PycharmProjects/IDS/dataset/CICIDS/file/preprocessed/'
    classification_m = 'binary'
    # classification_m = 'multi'

    # =========== TRAIN DATAFRAME PREPROCESSING ===========
    preprocess_df = BuildDataFrames(df_path=base_dir, df_type='train',
                                    classification_mode=classification_m)
    df = preprocess_df.ReadDataFrames()
    train, test = train_test_split(df, test_size=0.2)
    del df
    del preprocess_df

    preprocess_train = BuildDataFrames(dataframe=train, df_type='train', classification_mode=classification_m)
    train_tosave, min_max_scaler_obj = preprocess_train.Normalization()

    # =========== TEST DATAFRAME PREPROCESSING ===========
    preprocess_test = BuildDataFrames(dataframe=test, df_type='test', classification_mode=classification_m)
    test_tosave = preprocess_test.Normalization(min_max_scaler_object=min_max_scaler_obj)

    # # =========== FEATURE (COLUMN) EQUALIZATION BETWEEN TRAIN AND TEST DATAFRAMES  ===========
    # train_rescaled = preprocess_train.RescalingBetweenTrainTest(train_one_hot_cols=train_one_hot_cols,
    #                                                             test_one_hot_cols=test_one_hot_cols)
    # test_rescaled = preprocess_test.RescalingBetweenTrainTest(train_one_hot_cols=train_one_hot_cols,
    #                                                           test_one_hot_cols=test_one_hot_cols)

    # # =========== CHECKING ===========
    # diff_test_train = list(set(test_rescaled.columns) - set(train_rescaled.columns))
    # diff_train_test = list(set(train_rescaled.columns) - set(test_rescaled.columns))
    # print('CHECKING => these should be two EMPTY lists:', diff_train_test, diff_test_train)

    # train_temp, test_temp = train_rescaled, test_rescaled

    # =========== LABEL BINARIZE FOR MULTI-CLASSIFICATION MODE ===========
    if classification_m == 'multi':
        train_tosave, LabelBinerizerObj = preprocess_train.LabelBinarize()
        test_tosave = preprocess_test.LabelBinarize(LabelBinarizerObj=LabelBinerizerObj)

    # # =========== COLUMNS ORDER EQUALIZATION FOR FURTHER PICTURE FORMATTING ===========
    # train_sorted, test_sorted = SortColumnsBetweenTrainTest(train_temp, test_temp)

    # =========== CREATE 2D PICTURE ARRAYS FROM DATAFRAMES (SIMPLE METHOD) ===========
    # X_train, y_train = PictureFormat(train_sorted, classification_m)
    # print('=============================')
    # X_test, y_test = PictureFormat(test_sorted, classification_m)

    # =========== SAVE RESULTS ===========
    SaveDataFrames(train_tosave, 'train', classification_m, save_path)
    SaveDataFrames(test_tosave, 'test', classification_m, save_path)

    # SaveArray(X_train, y_train, 'train')
    # SaveArray(X_test, y_test, 'test')

    X, y = parse_data(train_tosave, dataset_name='CICIDS', classification_mode=classification_m)
    print(X.shape, y.shape)

    X, y = parse_data(test_tosave, dataset_name='CICIDS', classification_mode=classification_m)
    print(X.shape, y.shape)
