from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from datetime import datetime

class Data():
    def __init__(self, data, target_column,
                 test_size=0.15, val_size=0.05,
                 random_seed=42, batch_size = 16,
                    scaler_name='salt_adsorption',
                 use_cross_validation=False):
        self.data = data
        self.target_column = target_column
        self.random_seed = random_seed
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.use_cross_validation=use_cross_validation
        self.scaler_name = scaler_name

    def __get__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def preprocess(self, method='GAN', load_scaler=False, save_scaler=False, scaler_path=None):
        # Get features (X) and target (y)
        X = self.data.drop(self.target_column, axis=1)
        self.y = self.data[self.target_column].values #.reshape(-1,1)
        logger.info(f'mean of Y: {np.mean(self.y)}')
        # Standardizing data
        logger.info('Initiated Standardizing data')
        if load_scaler is True:
            with open(f"./models/{method}/{scaler_path}", 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler()
            scaler.fit(X)
        self.X = scaler.transform(X)
        # save scaler
        if load_scaler is False and save_scaler is True:
            with open(f'./models/{method}/{datetime.now().strftime("%Y-%m-%d")}_{self.scaler_name}_scaler_seed{self.random_seed}_train_size{(1 - self.test_size - self.val_size)*100}percent.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        logger.success('Standardizing completed')
        
        # train-test split for model evaluation and validation: (1 - val_size - test_size) train, test_size test, val_size validation
        logger.info('Initiated train-test split for model evaluation and validation')
        if self.use_cross_validation is False:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=(self.test_size+self.val_size), random_state=self.random_seed, shuffle=True)
            self.X_test, self.X_valid, self.y_test, self.y_valid = train_test_split(self.X_test, self.y_test, test_size=self.test_size / (self.test_size+self.val_size), random_state=self.random_seed, shuffle=True)

            logger.success(f'Train: {self.X_train.shape, self.y_train.shape} --> {self.y_train.shape[0] / (self.y_train.shape[0] + self.y_test.shape[0] + self.y_valid.shape[0]) * 100:.2f}%')
            logger.success(f'Val: {self.X_valid.shape, self.y_valid.shape} --> {self.y_valid.shape[0] / (self.y_train.shape[0] + self.y_test.shape[0] + self.y_valid.shape[0]) * 100:.2f}%')
            logger.success(f'Test: {self.X_test.shape, self.y_test.shape} --> {self.y_test.shape[0] / (self.y_train.shape[0] + self.y_test.shape[0] + self.y_valid.shape[0]) * 100:.2f}%')
            
            logger.success('train-test split completed')
        
            return self.X_train, self.X_test, self.X_valid, self.y_train, self.y_test, self.y_valid
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed, shuffle=True)

            logger.success(f'Train: {self.X_train.shape, self.y_train.shape} --> {self.y_train.shape[0] / (self.y_train.shape[0] + self.y_test.shape[0]) * 100:.2f}%')
            logger.success(f'Test: {self.X_test.shape, self.y_test.shape} --> {self.y_test.shape[0] / (self.y_train.shape[0] + self.y_test.shape[0]) * 100:.2f}%')

            logger.success('train-test split completed')
    
            return self.X_train, self.X_test, self.y_train, self.y_test


class Imputation():
    def __init__(self,
                 model,
                 method,
                    data: pd.DataFrame, 
                 columns: list):
        self.data = data
        self.columns = columns
        self.model = model
        self.method = method


    def impute(self, model, data, columns, method):
        """
        Fill missing values in a dataset using MICE imputation technique.
        
        Args:
            model: The imputation model to use (e.g., MICE).
            method: The name of the Machine Learning model (e.g., 'Bayesian', 'etree').
        
        Returns:
            A pandas DataFrame with missing values in 'columns' imputed.
        """
        logger.info(f'Database path: {data.head()}')
        logger.info(f'ML method: {method}\n')
        db_size = len(data)
        logger.info(f'Number of data points: {db_size}')
        for col in columns:
            missing_cnt = data[col].isnull().sum()
            logger.warning(f'Number of missing values in "{col}": {missing_cnt}')
        
        # Convert specified columns to numpy array for imputation
        data_to_impute = data[columns].to_numpy()
        
        # Copy the data for imputation
        imputation_data = data_to_impute.copy()
        
        # Fit and transform the data using the specified model
        imputed_data = model.fit_transform(imputation_data)
        
        # Convert the imputed data back to a pandas DataFrame
        imputed_df = pd.DataFrame(data=imputed_data, columns=self.columns)

        logger.success(f'Final Data shape is {imputed_df.shape}.')
        logger.success(f'Imputation completed using {method} model.')
        
        return imputed_df
    
    def impute_data(self, split_data=False):

        if split_data:

            # Groups
            group1 = ['SSA', 'PV', 'Psave', 'PVmicro', 'ID/IG', 'N', 'O']
            group2 = ['VW', 'FR', 'CNaCl', 'EC']

            # corresponding data
            group_1_data = self.data[group1]
            group_2_data = self.data[group2]
            
            filled_group_1_data = self.impute(self.model, group_1_data, group1, self.method)
            filled_group_2_data = self.impute(self.model, group_2_data, group2,self.method)

            # merge data
            df_imputed = pd.concat([filled_group_1_data, filled_group_2_data], axis=1)

        else:
            df_imputed = self.impute(self.model, self.data, self.columns, self.method)

        return df_imputed
    

import pacmap
import hdbscan

def cluster_data(data, 
                 scaler = None, 
                 dr_model = None, 
                 cl_model = None, 
                 train_data = None):
        """
        It takes in a dataframe, scales it, projects it into 2D, and then clusters it
        
        :param data: the dataframe to be clustered
        :param scaler: the scaler used to scale the data. If None, a new scaler is created
        :param dr_model: the dimensionality reduction model
        :param cl_model: the clustering model
        :param train_data: the dataframe containing the training data
        :return: A dictionary with the clustered data, the scaler, the DR model and the CL model.
        """
        
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(-1, 1))
        
        scaled_data = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-1]))

        logger.info(f'Simulated data: {scaled_data.shape}')
        scaled_data.head()

        # Project data in 2D
        if dr_model is None:
            logger.info('Performing DR...')
            dr_model = pacmap.PaCMAP(
                n_neighbors=10,
                #MN_ratio=5,
                #FP_ratio=20,
                # distance=euclidean,
                lr=1.0,
                apply_pca=True,
                verbose=False,
                intermediate=False,
                save_tree=True
                )

            reduced_data = dr_model.fit_transform(scaled_data.values, )

        else:
            logger.info(f'Using DR model provided')
            reduced_data = dr_model.transform(scaled_data.values, basis = train_data.iloc[:, :-1].values)

        # get clusters
        if cl_model is None:
            logger.info('Clustering...')
            cl_model = hdbscan.HDBSCAN(
                min_samples=50)#int(np.floor(scaled_data.shape[0] / 50)))
            labels = cl_model.fit_predict(reduced_data)
        
        else:
            logger.info(f'Using CL model provided')
            labels = cl_model.fit_predict(reduced_data)

        logger.info(f'd0: {reduced_data[:, 0].shape}')
        logger.info(f'd1: {reduced_data[:, 1].shape}')
        #logger.info(f'd2: {reduced_data[:, 2].shape}')
        logger.info(f'labels: {labels.shape}')
        logger.info(f'Unique labels: {len(set(labels))}')

        clustered_df = pd.DataFrame(
            data={
                'd0': reduced_data[:, 0],
                'd1': reduced_data[:, 1],
                # 'd2': reduced_data[:, 2],
                'label': labels
            })

        output = {
            'clustered_df': clustered_df,
            'cluster_scaler': scaler,
            'dr_model': dr_model,
            'cl_model': cl_model
        }

        return output