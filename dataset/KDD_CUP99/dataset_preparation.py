import pandas as pd
import numpy as np
import os

from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from pathlib import Path

from utils import parse_data, save_dataframe, sort_columns

np.random.seed(0)


class BuildDataFrames:
    def __init__(self, train_path: str, test_path: str, normalization_method: str, classification_mode: str = 'binary'):
        self.test = None
        self.train = None
        self.train_path = train_path
        self.test_path = test_path
        # self.augmentation = augmentation
        self.read_data_frames()
        self.classification_mode = classification_mode

    def read_data_frames(self):
        feature = ["duration", "protocol_type", "service", "flag", "src_bytes",
                   "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                   "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                   "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                   "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
        self.train = pd.read_csv(self.train_path, names=feature)
        self.test = pd.read_csv(self.test_path, names=feature)

        self.train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        self.test.drop(['num_outbound_cmds'], axis=1, inplace=True)
        return self.train, self.test

    def label_mapping(self):
        """
        this function specifically is used for original dataset
        """
        if self.classification_mode == 'multi':
            self.train.label.replace(['normal.'], 'normal', inplace=True)
            self.train.label.replace(
                ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'Dos', inplace=True)
            self.train.label.replace(
                ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'R2L', inplace=True)
            self.train.label.replace(
                ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'Probe', inplace=True)
            self.train.label.replace(
                ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'],
                'U2R', inplace=True)

            self.test.label.replace(['normal.'], 'normal', inplace=True)
            self.test.label.replace(
                ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.'],
                'Dos', inplace=True)
            self.test.label.replace(
                ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'],
                'R2L', inplace=True)
            self.test.label.replace(
                ['ipsweep.', 'nmap.', 'portsweep.', 'satan.'],
                'Probe', inplace=True)
            self.test.label.replace(
                ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'],
                'U2R',
                inplace=True)
        elif self.classification_mode == 'binary':
            self.train['label'] = self.train['label'].apply(lambda x: 'attack.' if x != 'normal.' else x)
            self.test['label'] = self.test['label'].apply(lambda x: 'attack.' if x != 'normal.' else x)
        return self.train, self.test

    def normalization(self, normalization_method):
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

    def label_binarizing(self):
        if self.classification_mode == 'multi':
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

        elif self.classification_mode == 'binary':
            label_mapping = {'normal.': 0, 'attack.': 1}
            self.train['label'] = self.train['label'].map(label_mapping)
            self.test['label'] = self.test['label'].map(label_mapping)
        return self.train, self.test

    def smote(self, label_feature_name):
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

    def smoteenn(self, label_feature_name):  # first do one-hot encoding
        # column_list = list(self.train.columns)
        # feature_list = column_list.remove(label_feature_name)
        data = self.train.drop([label_feature_name], axis=1)
        label = self.train[label_feature_name]
        smoteenn = SMOTEENN(random_state=42, enn=EditedNearestNeighbours(sampling_strategy='auto'))
        X_SENN_aug, y_SENN_aug = smoteenn.fit_resample(data, label)
        self.train = pd.concat([X_SENN_aug, y_SENN_aug], axis=1)
        return self.train

    def save_data_frames(self, output_path):
        train_file_name = 'train'
        test_file_name = 'test'
        if self.classification_mode == 'binary':
            train_file_name = train_file_name + '_binary'
            test_file_name = test_file_name + '_binary'
        elif self.classification_mode == 'multi':
            train_file_name = train_file_name + '_multi'
            test_file_name = test_file_name + '_multi'
        if output_path is not None:
            train_file = os.path.join(output_path, train_file_name + '.csv')
            test_file = os.path.join(output_path, test_file_name + '.csv')
            self.train.to_csv(train_file, index=False)
            self.test.to_csv(test_file, index=False)
            print('Saved:', train_file, test_file)

    def get_data_frames(self):
        return self.train, self.test


if __name__ == "__main__":
    # preprocess Object is for creating normal dataset (reading, label mapping, normalization, one-hot encoding with
    # rescale train and test set, label binarizing)

    base_path = Path(__file__).resolve().parent.joinpath('file')
    train_path = base_path.joinpath('original', 'kddcup.data_10_percent_corrected')
    test_path = base_path.joinpath('original', 'corrected.gz')
    save_path = base_path.joinpath('preprocessed')
    # classification_mode = 'binary'
    classification_mode = 'multi'

    preprocess = BuildDataFrames(train_path=str(train_path), test_path=str(test_path),
                                 normalization_method='normalization',
                                 classification_mode=classification_mode)

    preprocess.label_mapping()
    normalized_train, normalized_test = preprocess.normalization(normalization_method='normalization')
    onehot_train, onehot_test = preprocess.one_hot_encoding(label_feature_name='label')
    # preprocess.smoteenn('label')
    label_binarized_train, label_binarized_test = preprocess.label_binarizing()
    preprocess.save_data_frames(save_path)
    train, test = preprocess.get_data_frames()

    X, y = parse_data(train, dataset_name='KDD_CUP99', classification_mode=classification_mode)
    print(f'train shape: x=>{X.shape}, y=>{y.shape}')

    X, y = parse_data(test, dataset_name='KDD_CUP99', classification_mode=classification_mode)
    print(f'test shape: x=>{X.shape}, y=>{y.shape}')
