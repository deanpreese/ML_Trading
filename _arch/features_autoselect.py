import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel

from common_func import gen_importances, show_stats, calc_MSE


from RFTModel import RFTModel
from XtraModel import XtraModel
from GBRModel import GBRModel
from MLPRModel import MLPRModel
from LTSMModel import LTSMModel
from MLPRTFModel import MLPRTFModel
from CBRModel import CBRModel
from XGBModel import XGBModel



import warnings
warnings.filterwarnings("ignore")

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
        "data/Seq_buildSeqInd_R3070_DIFF_X.csv",

]


from os import system
clear = lambda: system('clear')

for filename in datafiles:
    
    data = pd.read_csv(filename)   
        
    max_runs = 50
    run_batch_size = 32
    run_test_size = 0.8

    num_columns = len(data.axes[1]) 
    input_features =  num_columns -1
    X = data.iloc[:, 0:input_features]  
    y = data['output'].values

    # Split the dataset into training and testing sets
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=run_test_size, random_state=0)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_1 = scaler.fit_transform(X_train_1)
    X_test_1  = scaler.transform(X_test_1)

    BaseModel = RFTModel()
    model = BaseModel.model
    model.fit(X_train_1, y_train_1)
    predicted_base_values = model.predict(X_test_1)
    
    print(" ")
    print(f"Baseline MSE && RMSE ")
    mse, rmse = calc_MSE(y_test_1, predicted_base_values, True)
   
    baseline_mse = mse
    baseline_rmse = rmse
    best_mse = mse
    best_rmse = rmse
    best_features= []
    best_correct = 0
    best_total = 0
        
    for featurecount in (range(10, 100, 10)):
        
        mx_feat = int(input_features * (featurecount/100))
        
        #select = SelectFromModel(model, threshold=0.03, max_features=mx_feat)
        select = SelectFromModel(model, max_features=mx_feat)

        select.fit_transform(X_train_1, y_train_1)
        cols_idxs = select.get_support(indices=True)

        selected_features_data = X.iloc[:, cols_idxs]
        X_train, X_test, y_train, y_test = train_test_split(selected_features_data, y, test_size=run_test_size, random_state=0)

        if X_train.shape[1] < 1 :
            print("No Features selected")
            exit()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        #model_2 = RFTModel()
        model_2 = XGBModel()
        model_2.load_model_data(input_features, X_train, y_train, X_test, y_test )
                
        print("Feature Scaled Model")
        print(f"{model_2.model}")
        
        model_2.train_model()
        predicted_values = model_2.predict_model()

        print(f"MAX Number of Features Used:  {mx_feat} ")
        
        mse = mean_squared_error(y_test, predicted_values)
        rmse = mse**.5
        if mse < best_mse :
            best_features.clear()
            best_correct, best_total = show_stats(False, y_test, predicted_values)
            best_features = ShowImportances(data.columns[:input_features],model_2.get_importances(), False)
            best_mse = mse
            best_rmse = rmse
            
        """
        print(f"MSE  {mse}")
        print(f"RMSE {rmse}")
        print(f"Correct {best_correct}")
        print(f"Total   {best_total}")
        print(" ")
        print(f"{best_features}")
        #print(f"{selected_features_data} ")
        print(" ")
        """
        model = model_2.model
    
    
    
    print(" ")
    print(" ----------------- ")
    print(f"Baseline MSE {baseline_mse}") 
    print(f"Baseline RMSE {baseline_rmse}") 
    print(" ")
    print("Best Model ")
    print(f"Using {len(best_features)} Features")
    print(f"MSE  {best_mse}")
    print(f"RMSE {best_rmse}")
    print(" ")
    print(f"Correct {best_correct}")
    print(f"Total   {best_total}")
    print(" ")
    print(f"{best_features}")
    print(" ")
