import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import (MinMaxScaler, LabelBinarizer)
from utils import parse_data


class BuildDataFrames:

    def __init__(self, df_path: str, df_type: str = 'train', classification_mode: str = 'multi'):

        self.DataFrame = None
        self.df_path = df_path
        self.df_type = df_type
        self.classification_mode = classification_mode
        self.label_feature_name = 'attack_cat' if self.classification_mode == 'multi' else 'label'

    def ReadDataFrames(self):

        cols_to_drop = ['id', 'label'] if self.classification_mode == 'multi' else ['id', 'attack_cat']
        self.DataFrame = pd.read_csv(self.df_path)
        self.DataFrame.drop(cols_to_drop, axis=1, inplace=True)
        return self.DataFrame

    def Normalization(self, min_max_scaler_object=None):

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

    def OneHotEncoding(self):

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

    def RescalingBetweenTrainTest(self, train_one_hot_cols, test_one_hot_cols):

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

    def LabelBinarize(self, LabelBinarizerObj=None):

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

            TestBinarizedLabel = LabelBinarizerObj.transform(self.DataFrame[self.label_feature_name])
            TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel, columns=LabelBinarizerObj.classes_)
            self.DataFrame = pd.concat(
                [self.DataFrame.drop([self.label_feature_name], axis=1), TestBinarizedLabelDataFrame],
                axis=1)
            return self.DataFrame


def SortColumnsBetweenTrainTest(train_df, test_df):
    train_cols = train_df.columns
    test_sortedBasedOnTrain = pd.DataFrame(columns=train_cols)
    for col in test_sortedBasedOnTrain:
        test_sortedBasedOnTrain[col] = test_df[col]

    return train_df, test_sortedBasedOnTrain


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


def SaveDataFrames(DataFrame, DataFrameType, classification_mode):
    save_path = '/home/faraz/PycharmProjects/IDS/dataset/UNSW_NB15/file/preprocessed/'
    file_name = DataFrameType
    if classification_mode == 'binary':
        file_name = file_name + '_binary'
    elif classification_mode == 'multi':
        file_name = file_name + '_multi'
    train_file = os.path.join(save_path, file_name + '.csv')
    DataFrame.to_csv(train_file, index=False)

    print('Saved:', train_file)


def SaveArray(X, y, ArrayType):
    X_name = 'X_' + ArrayType + '.npy'
    np.save(X_name, X)
    y_name = 'y_' + ArrayType + '.npy'
    np.save(y_name, y)


if __name__ == "__main__":
    train_path = '/home/faraz/PycharmProjects/IDS/dataset/UNSW_NB15/file/original/UNSW_NB15_training-set.csv'
    test_path = '/home/faraz/PycharmProjects/IDS/dataset/UNSW_NB15/file/original/UNSW_NB15_testing-set.csv'
    classification_m = 'binary'
    # classification_m = 'multi'

    # =========== TRAIN DATAFRAME PREPROCESSING ===========
    preprocess_train = BuildDataFrames(df_path=train_path, df_type='train',
                                       classification_mode=classification_m)
    train = preprocess_train.ReadDataFrames()
    train_normalized, min_max_scaler_obj = preprocess_train.Normalization()
    train_one_hot, train_one_hot_cols = preprocess_train.OneHotEncoding()

    # =========== TEST DATAFRAME PREPROCESSING ===========
    preprocess_test = BuildDataFrames(df_path=test_path, df_type='test',
                                      classification_mode=classification_m)
    test = preprocess_test.ReadDataFrames()
    test_normalized = preprocess_test.Normalization(min_max_scaler_object=min_max_scaler_obj)
    test_one_hot, test_one_hot_cols = preprocess_test.OneHotEncoding()

    # =========== FEATURE (COLUMN) EQUALIZATION BETWEEN TRAIN AND TEST DATAFRAMES  ===========
    train_rescaled = preprocess_train.RescalingBetweenTrainTest(train_one_hot_cols=train_one_hot_cols,
                                                                test_one_hot_cols=test_one_hot_cols)
    test_rescaled = preprocess_test.RescalingBetweenTrainTest(train_one_hot_cols=train_one_hot_cols,
                                                              test_one_hot_cols=test_one_hot_cols)

    # =========== CHECKING ===========
    diff_test_train = list(set(test_rescaled.columns) - set(train_rescaled.columns))
    diff_train_test = list(set(train_rescaled.columns) - set(test_rescaled.columns))
    print('CHECKING => these should be two EMPTY lists:', diff_train_test, diff_test_train)

    train_temp, test_temp = train_rescaled, test_rescaled

    # =========== LABEL BINARIZE FOR MULTI-CLASSIFICATION MODE ===========
    if classification_m == 'multi':
        train_LabelBinerized, LabelBinerizerObj = preprocess_train.LabelBinarize()
        test_LabelBinerized = preprocess_test.LabelBinarize(LabelBinarizerObj=LabelBinerizerObj)
        train_temp, test_temp = train_LabelBinerized, test_LabelBinerized

    # =========== COLUMNS ORDER EQUALIZATION FOR FURTHER PICTURE FORMATTING ===========
    train_sorted, test_sorted = SortColumnsBetweenTrainTest(train_temp, test_temp)

    # =========== CREATE 2D PICTURE ARRAYS FROM DATAFRAMES (SIMPLE METHOD) ===========
    # X_train, y_train = PictureFormat(train_sorted, classification_m)
    # print('=============================')
    # X_test, y_test = PictureFormat(test_sorted, classification_m)

    # =========== SAVE RESULTS ===========
    SaveDataFrames(train_sorted, 'train', classification_m)
    SaveDataFrames(test_sorted, 'test', classification_m)

    # SaveArray(X_train, y_train, 'train')
    # SaveArray(X_test, y_test, 'test')

    X, y = parse_data(train_sorted, dataset_name='UNSW_NB15', classification_mode=classification_m)
    print(X.shape, y.shape)
