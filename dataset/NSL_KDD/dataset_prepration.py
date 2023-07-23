import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(0)


class BuildDataFrames:
    def __init__(self, train_path: str, test_path: str, normalization_method: str):
        self.test = None
        self.train = None
        self.train_path = train_path
        self.test_path = test_path
        # self.augmentation = augmentation
        self.read_data_frames()
        '''
        if augmentation == 0:
          self.LabelMapping()
          self.Normalization( normalization_method)
          self.OneHotEncoding( label_feature_name= 'label')
          self.LabelBinarize()
          self.SaveDataFrames(version)
        else:
          self.LabelMapping()
          self.Normalization( normalization_method)
          self.AugmentationSMOTE('label')
          self.OneHotEncoding( label_feature_name= 'label')
          self.LabelBinarize()
          self.SaveDataFrames(version)
        '''

    def read_data_frames(self):
        feature = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
                   "urgent", "hot",
                   "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells",
                   "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
                   "serror_rate", "srv_serror_rate",
                   "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                   "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                   "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
                   "difficulty"]
        self.train = pd.read_csv(self.train_path, names=feature)
        self.test = pd.read_csv(self.test_path, names=feature)

        self.train.drop(['difficulty', 'num_outbound_cmds'], axis=1, inplace=True)
        self.test.drop(['difficulty', 'num_outbound_cmds'], axis=1, inplace=True)
        return self.train, self.test

    def label_mapping(self):
        """
        this function specificly is used for NSL-KDD dataset
        """

        self.train.label.replace(
            ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'apache2', 'processtable', 'udpstorm'],
            'Dos', inplace=True)
        self.train.label.replace(
            ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'sendmail',
             'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm'], 'R2L', inplace=True)
        self.train.label.replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe', inplace=True)
        self.train.label.replace(
            ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel'], 'U2R',
            inplace=True)

        self.test.label.replace(
            ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'apache2', 'processtable', 'udpstorm'],
            'Dos', inplace=True)
        self.test.label.replace(
            ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'sendmail',
             'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm'], 'R2L', inplace=True)
        self.test.label.replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe', inplace=True)
        self.test.label.replace(
            ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel'], 'U2R',
            inplace=True)
        return self.train, self.test

    def normalization(self, normalization_method):
        # self.train, self.test = self.ReadDataFrames()
        """
        this function takes 2 dataframe (train, test) and return normalized (train, test) by 2 method:
        normalization & standardization
        """
        if normalization_method == 'normalization':
            train_df = self.train
            numeric_col_list = train_df.select_dtypes(include='number').columns
            train_df_numeric_col = train_df[numeric_col_list].values
            min_max_scaler = MinMaxScaler()
            train_df_numeric_col_scaled = min_max_scaler.fit_transform(train_df_numeric_col)
            train_df_numeric_col_scaled = pd.DataFrame(train_df_numeric_col_scaled, columns=numeric_col_list)
            self.train[numeric_col_list] = train_df_numeric_col_scaled

            test_df = self.test
            test_df_numeric_col = test_df[numeric_col_list].values
            test_df_numeric_col_scaled = min_max_scaler.transform(test_df_numeric_col)
            test_df_numeric_col_scaled = pd.DataFrame(test_df_numeric_col_scaled, columns=numeric_col_list)
            self.test[numeric_col_list] = test_df_numeric_col_scaled
            return self.train, self.test

        elif normalization_method == 'standardization':
            std_scaler = StandardScaler()
            # train_columns = self.train.columns
            train_numeric_columns = self.train.select_dtypes(include='number').columns
            for col in train_numeric_columns:
                arr = self.train[col]
                arr = np.array(arr)
                self.train[col] = std_scaler.fit_transform(arr.reshape(len(arr), 1))

            test_numeric_columns = self.test.select_dtypes(include='number').columns
            for col in test_numeric_columns:
                arr = self.test[col]
                arr = np.array(arr)
                self.test[col] = std_scaler.transform(arr.reshape(len(arr), 1))

            return self.train, self.test

    # def one_hot_encoding(self, label_feature_name):  # label_feature_name for NSL-KDD is 'label'
    #     categorical_column_list = list(self.train.select_dtypes(include='object').columns)
    #     categorical_column_list.remove(label_feature_name)
    #
    #     one_hot_train = pd.get_dummies(self.train, columns=categorical_column_list)
    #     one_hot_test = pd.get_dummies(self.test, columns=categorical_column_list)
    #
    #     # check if test and train don't have same features after onehot-encoding, so upcoming lines of code of this
    #     # function, rescale test and train dataset
    #     df_test = pd.DataFrame(data=None, columns=one_hot_train.columns)
    #     zero_values = [0] * len(one_hot_test)
    #     for c in df_test.columns:
    #         if c not in list(one_hot_test.columns):
    #             df_test[str(c)] = zero_values
    #         else:
    #             df_test[str(c)] = one_hot_test[str(c)]
    #
    #     train_label_col = one_hot_train.pop(label_feature_name)
    #     # position parameter for insert function starts with 0 so we don't need to
    #     # len(one_hot_train.columns)+1 (after pop function)
    #     one_hot_train.insert(len(one_hot_train.columns), label_feature_name,
    #                          train_label_col)
    #
    #     test_label_col = df_test.pop(label_feature_name)
    #     df_test.insert(len(df_test.columns), label_feature_name, test_label_col)
    #
    #     self.train = one_hot_train
    #     self.test = df_test
    #
    #     return self.train, self.test

    def one_hot_encoding(self, label_feature_name):
        categorical_column_list = list(self.train.select_dtypes(include='object').columns)
        categorical_column_list.remove(label_feature_name)

        one_hot_train = pd.get_dummies(self.train, columns=categorical_column_list)
        one_hot_test = pd.get_dummies(self.test, columns=categorical_column_list)

        # Combine one-hot encoded columns to make sure train and test have the same features
        all_columns = set(one_hot_train.columns) | set(one_hot_test.columns)
        df_train = pd.concat([one_hot_train, pd.DataFrame(columns=all_columns)]).fillna(0)
        df_test = pd.concat([one_hot_test, pd.DataFrame(columns=all_columns)]).fillna(0)

        train_label_col = df_train.pop(label_feature_name)
        df_train.insert(len(df_train.columns), label_feature_name, train_label_col)

        test_label_col = df_test.pop(label_feature_name)
        df_test.insert(len(df_test.columns), label_feature_name, test_label_col)

        self.train = df_train
        self.test = df_test

        return self.train, self.test

    def label_binarize(self):
        # create an object of label binarizer, then fit on train labels
        LabelBinarizerObject_fittedOnTrainLabel = LabelBinarizer().fit(self.train['label'])
        # transform train labels with that object
        TrainBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(self.train['label'])
        # convert transformed labels to dataframe
        TrainBinarizedLabelDataFrame = pd.DataFrame(TrainBinarizedLabel,
                                                    columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)
        # concatenate training set after drop 'label' with created dataframe of binarized labels
        self.train = pd.concat([self.train.drop(['label'], axis=1), TrainBinarizedLabelDataFrame], axis=1)

        TestBinarizedLabel = LabelBinarizerObject_fittedOnTrainLabel.transform(self.test['label'])
        TestBinarizedLabelDataFrame = pd.DataFrame(TestBinarizedLabel,
                                                   columns=LabelBinarizerObject_fittedOnTrainLabel.classes_)
        self.test = pd.concat([self.test.drop(['label'], axis=1), TestBinarizedLabelDataFrame], axis=1)
        return self.train, self.test

    def augmentation_smote(self, label_feature_name):
        categorical_column_list = list(self.train.select_dtypes(include='object').columns)
        categorical_column_list.remove(label_feature_name)
        cat_index = []
        for cat in categorical_column_list:
            cat_index.append(self.train.columns.get_loc(cat))
        smotenc = SMOTENC(random_state=123, categorical_features=cat_index)
        X_aug, y_aug = smotenc.fit_resample(self.train.drop([label_feature_name], axis=1),
                                            self.train[label_feature_name])
        self.train = pd.concat([X_aug, y_aug], axis=1)
        return self.train

    def smoteenn(self, label_feature_name):  # first do onehot encoding
        column_list = list(self.train.columns)
        feature_list = column_list.remove(label_feature_name)
        data = self.train.drop([label_feature_name], axis=1)
        label = self.train[label_feature_name]
        smoteenn = SMOTEENN(random_state=42, enn=EditedNearestNeighbours(sampling_strategy='auto'))
        X_SENN_aug, y_SENN_aug = smoteenn.fit_resample(data, label)
        self.train = pd.concat([X_SENN_aug, y_SENN_aug], axis=1)
        return self.train

    def picture_format(self):  # do picture_format after label_binarize
        class_columns = ['Dos', 'Probe', 'R2L', 'U2R', 'normal']

        # for train set
        X_train = self.train.drop(class_columns, axis=1)
        y_train = self.train[class_columns]

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print('X_train shape is:', np.shape(X_train), '\ny_train shape is:', np.shape(y_train))

        X_train = np.reshape(X_train, (X_train.shape[0], 11, 11, 1))
        print('X_train shape is:', np.shape(X_train), '\ny_train shape is:', np.shape(y_train))

        # for test set
        X_test = self.test.drop(class_columns, axis=1)
        y_test = self.test[class_columns]

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print('X_test shape is:', np.shape(X_test), '\ny_test shape is:', np.shape(y_test))

        X_test = np.reshape(X_test, (X_test.shape[0], 11, 11, 1))
        print('X_test shape is:', np.shape(X_test), '\ny_test shape is:', np.shape(y_test))

        return X_train, y_train, X_test, y_test

    def save_data_frames(self, output_path):
        if output_path is not None:
            train_file = os.path.join(output_path, 'train.csv')
            test_file = os.path.join(output_path, 'test.csv')
            self.train.to_csv(train_file, index=False)
            self.test.to_csv(test_file, index=False)
            print('Saved:', train_file, test_file)

    def get_data_frames(self):
        return self.train, self.test


if __name__ == "__main__":
    # preprocess Object is for creating normal dataset (reading, label mapping, normalization, onehot encoding with
    # rescale train and test set, label binarize)

    train_path = '/home/faraz/PycharmProjects/IDS/dataset/NSL_KDD/file/NSL-KDD/KDDTrain+.txt'
    test_path = '/home/faraz/PycharmProjects/IDS/dataset/NSL_KDD/file/NSL-KDD/KDDTest+.txt'
    test21_path = '/home/faraz/PycharmProjects/IDS/dataset/NSL_KDD/file/NSL-KDD/KDDTest-21.txt'
    save_path = '/home/faraz/PycharmProjects/IDS/dataset/NSL_KDD/file/preprocessed'

    preprocess = BuildDataFrames(train_path=train_path, test_path=test_path, normalization_method='normalization')

    preprocess.label_mapping()
    normalized_train, normalized_test = preprocess.normalization(normalization_method='normalization')
    onehot_train, onehot_test = preprocess.one_hot_encoding(label_feature_name='label')
    label_binarized_train, label_binarized_test = preprocess.label_binarize()
    preprocess.save_data_frames(save_path)
