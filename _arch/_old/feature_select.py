import numpy as np
import pandas as pd
import pickle
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor ,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, explained_variance_score, mean_absolute_percentage_error,accuracy_score
from sklearn.feature_selection import SelectFromModel

from common_func import ShowImportances, show_stats, write_line_to_file

import warnings
warnings.filterwarnings("ignore")


model_results = []
features_used = []
rf_features_list = []    

def runMLP(input_features, X_train, y_train, X_test, y_test ):
        # -------------------
        #  MLP Regression
        # -------------------         
        hlayer0 = int(input_features * .9)
        hlayer1 = int(input_features * .75)
        hlayer2 = int(input_features * .66)
        hlayer3 = int(input_features * .5)
        hlayer4 = int(input_features * .25)

        params_mlp = { 'hidden_layer_sizes' :  [hlayer1,hlayer3,2,1],
            #'activation' : 'relu',
            #'solver' : 'adam',
            #'alpha' : 0.0,
            #'batch_size' : 32,
            'random_state' : 0,
            #'tol' : 0.000001,
            #'nesterovs_momentum' : False,
            #'learning_rate' : 'invscaling',
            #'learning_rate' : 'constant',
            #'learning_rate_init' : 0.001,
            'learning_rate_init' : 0.00088,
            'max_iter' : 250,
            #'shuffle' : False,
            #'early_stopping' : True,
            'n_iter_no_change' : 25,
            'verbose' : True }
        
        model_mlp = MLPRegressor(**params_mlp)        
        model_mlp.fit(X_train, y_train)
        predicted_mlp_values = model_mlp.predict(X_test)

        print(" ")
        print(f'{model_mlp}')
        print(" ")
        
        correct = show_stats(False, y_test, predicted_mlp_values)
        
        mse = mean_squared_error(y_test, predicted_mlp_values)
        rmse = mse**.5
        
        mlp_tuple = ("MLP",input_features+1, mse , rmse, correct, features_used, params_mlp)
        return (mlp_tuple)


def runGBR(input_features, X_train, y_train, X_test, y_test ):

        params_gbr = {
                'n_estimators': 700,
                "max_depth": 40,
                "min_samples_split": 5,
                "learning_rate": 0.01,
                'loss': 'squared_error',
                "min_samples_leaf" : 20,
                'random_state' : 0,
                'verbose' : 0,
                'max_features': 'log2',
                'n_iter_no_change' : 25
        }
        
        model_gbr = GradientBoostingRegressor(**params_gbr)

        model_gbr.fit(X_train, y_train)
        predicted_mlp_values = model_gbr.predict(X_test)

        print(" ")
        print(f'{model_gbr}')
        print(" ")
        
        correct = show_stats(False, y_test, predicted_mlp_values)
        
        mse = mean_squared_error(y_test, predicted_mlp_values)
        rmse = mse**.5
        
        gbr_tuple = ("GBR",input_features+1, mse , rmse, correct, features_used, params_gbr )
        return gbr_tuple


def runXtra(input_features, X_train, y_train, X_test, y_test ):

        params_x=   {'n_estimators': 100, 
                     'min_samples_split': 15, 
                     'min_samples_leaf': 1, 
                     'max_features': 'sqrt', 
                     'max_depth': 150,
                     'n_jobs' : 6,   
                     'verbose' : 0
                     }

        params_xtra = {
                'n_estimators': 100,
                #"max_depth": 50,
                #"min_samples_split": 5,
                #"learning_rate": 0.01,
                'criterion': 'squared_error',
                #"min_samples_leaf" : 1,
                'random_state' : 0,
                'max_features': 'sqrt', 
                'n_jobs' : 6,   
                'verbose' : 0
        }
        
        model_xtra = ExtraTreesRegressor(**params_x)

        model_xtra.fit(X_train, y_train)
        predicted_xtra_values = model_xtra.predict(X_test)

        print(" ")
        print(f'{model_xtra}')
        print(" ")
        
        correct = show_stats(False, y_test, predicted_xtra_values)
        
        mse = mean_squared_error(y_test, predicted_xtra_values)
        rmse = mse**.5
        
        xtra_tuple = ("XT",input_features+1, mse , rmse, correct, features_used , params_xtra)
        return xtra_tuple


def runFile(filetorun):

        max_runs = 50
        run_batch_size = 32
        run_test_size = 0.9

        data = pd.read_csv(filetorun) 

        num_columns = len(data.axes[1]) 
        input_features =  num_columns -1
        X = data.iloc[:, 0:input_features]  
        y = data['output'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------

        params_rfa = {
                'n_estimators': 100,
                #"max_depth": 30,
                #"min_samples_split": 20,
                #"learning_rate": 0.01,
                'criterion': 'squared_error',
                #"min_samples_leaf" : 50,
                'random_state' : 0,
                'max_features': 'sqrt', 
                'n_jobs' : 6,   
                'verbose' : 0
        }
        model = RandomForestRegressor(**params_rfa) 

        lpstop = 5
        lpstep = -2

        for targetCount in (range(num_columns, lpstop, lpstep)):


                print( " Model Results " )
                if useMLPR:
                        print( " -------" )
                        model_results.append(runMLP(input_features, X_train, y_train, X_test, y_test ))
                        
                if useGBR:                        
                        print( " -------" )
                        model_results.append(runGBR(input_features, X_train, y_train, X_test, y_test ))
                        
                if useXtra:        
                        print( " -------" )
                        model_results.append(runXtra(input_features, X_train, y_train, X_test, y_test ))
                        print( " -------" )
                # -------------------
                #  Feature Selection 
                # -------------------         
                features_used.clear()
                
                if targetCount < abs(lpstep):
                        break
                
                print( " Doing Feature Extraction" )
                print(" ")
                print(f'{model}')
                print(" ")
                        
                model.fit(X_train, y_train)
                predicted_values = model.predict(X_test)

                correct = show_stats(False, y_test, predicted_values)
                mse = mean_squared_error(y_test, predicted_values)
                rmse = mse**.5

                feature_importance = list(zip(data.columns[:input_features], model.feature_importances_))
                sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

                fcnt = 0
                for feature, weight in sorted_feature_importance:
                        #txt = f"Feature: {feature}, Weight: {weight}"
                        if fcnt <= targetCount:
                                #print(f"{txt}")
                                features_used.append( f'{feature}'  )
                        fcnt = fcnt +1
                
                features_used.append("output")
                
                tuple_element = (len(features_used), mse , rmse, correct, features_used )
                rf_features_list.append(tuple_element)
                
                data = pd.read_csv(filetorun) 
                data = data[features_used]        
                num_columns = len(data.axes[1]) 
                input_features =  num_columns -1
                X = data.iloc[:, 0:input_features]  
                y = data['output'].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                print( f"{len(features_used)}  Features Selected" )        
                print(" ")



        



def printResults(filex, idname):

        print(" ")
        print("Random Forest Results ")
        print(" ")

        for count, mse, rmse, correct, features in rf_features_list:
                txt = f"{count}   {mse}   {rmse}   {correct}" 
                print(txt) 
                
        print(" ")
        print("Model Results ")
        print(" ")
        for modelname, count, mse, rmse, correct, features, model_summary in model_results:

                txt = f"{idname} , {modelname}  , {count}   ,   {mse}   ,   {rmse}   ,   {correct}  , {features} , {model_summary} " 
                write_line_to_file(filex, txt)
                
                print(f"{modelname}   {count}      {mse}      {rmse}      {correct}  ") 






useMLPR = False
useGBR = False
useXtra = True

datafiles = [ 
        #"data/Seq_buildSeq30_R3070_TAG.csv",
        #"data/Seq_buildSeqInd_R2080_TAG.csv",
        #"data/Seq_buildSeqInd_R3070_TAG.csv",
        #"data/Seq_buildSeqTopSet_R2080_TAG.csv",
        #"data/Seq_buildSeqTopSet_R3070_TAG.csv",
        #"data/Seq_TopSetX_R2080_TAG.csv",
        #"data/Seq_TopSetX_R3070_TAG.csv",
        #"data/Seq_buildSeq30_R3070_X_TAG.csv",
        #"data/Seq_buildSeqInd_R3070_TAG.csv",
        "data/Seq_buildSeqInd_R3070_DIFF_X.csv"
]



for filename in datafiles:
        
        runFile(filename)
        
        name, _, ext = filename.partition(".")
        
        parts = name.split("/")
        newFile = f"{parts[0]}/Results_{parts[1]}.{ext}"
        
        printResults(newFile, name)

        model_results = []

        
        

               