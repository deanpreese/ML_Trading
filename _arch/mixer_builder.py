import pandas as pd
import numpy as np
import pickle
from pyparsing import CaselessKeyword
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from common_func import gen_importances, show_stats, calc_MSE


num_epocs = 5
run_batch_size = 16
run_test_size = 0.8


datafile = [ 
        "data/Seq_buildSeqInd_Lucky13_3070.csv",
        "data/Seq_buildSeqInd_13x_ALL_3M.csv",
        "data/13X_15_ALL.csv",
        "data/Seq_buildSeqIndX_ALL_DIFF_15_BIG.csv",
]

data = pd.read_csv(datafile[1])

num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  

#Rft Set
#idxs = [47, 49, 50, 51, 52, 56, 57, 60, 61, 62, 67]

#idxs = [35, 36, 48, 50, 51, 52, 56, 57, 62]
#X = X.iloc[:, idxs]
#input_features = len(X.axes[1])

y = data['output'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=run_test_size, random_state=0)

print("  ")
print(f"TRAIN: {X_train.shape[0]}   {X_train.shape[1]}")
print(f"TEST: {X_test.shape[0]}  {X_test.shape[1]} ")
print(" ")

# Scale the features to the range [-1, 1]
#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

gbr = GBRModel()
gbr.load_model_data(input_features,X_train, y_train, X_test, y_test )
gbr.train_model()
gbr_predictions = gbr.predict_model()
gbr_perf, grb_tot = gbr.stats(True)
gbr_mse, gbr_rmse = gbr.MSE_RMSE(True)

xtra = XtraModel()
xtra.load_model_data(input_features,X_train, y_train, X_test, y_test )
xtra.train_model()
xtra_predictions = xtra.predict_model()
xtra_mse, xtra_rmse = xtra.MSE_RMSE(False)
xtra_perf, xtra_tot = xtra.stats(False)

mlpr = MLPRModel(input_features)
mlpr.load_model_data(input_features,X_train, y_train, X_test, y_test )
mlpr.train_model()
mlpr_predictions = mlpr.predict_model()
mlpr_mse, mlpr_rmse = mlpr.MSE_RMSE(False)
mlpr_perf, mlpr_tot = mlpr.stats(False)

rft = RFTModel()
rft.load_model_data(input_features,X_train, y_train, X_test, y_test )
rft.train_model()
rft_predictions = rft.predict_model()
rft_mse, rft_rmse = rft.MSE_RMSE(False)
rft_perf, rft_tot = rft.stats(False)

xgb = XGBModel()
xgb.load_model_data(input_features,X_train, y_train, X_test, y_test )
xgb.train_model()
xgb_predictions = xgb.predict_model()
xgb_mse, xgb_rmse = xgb.MSE_RMSE(False)
xgb_perf, xgb_tot = xgb.stats(False)

cbr = CBRModel()
cbr.load_model_data(input_features,X_train, y_train, X_test, y_test )
cbr.train_model()
cbr_predictions = cbr.predict_model()
cbr_perf, cbr_tot = cbr.stats(False)
cbr_mse, cbr_rmse = cbr.MSE_RMSE(False)

gbr_score = gbr.model.score(X_test, y_test)
xtra_score = xtra.model.score(X_test, y_test)
mlpr_score = mlpr.model.score(X_test, y_test)
rft_score = rft.model.score(X_test, y_test)
cbr_score = cbr.model.score(X_test, y_test)
xgb_score = xgb.model.score(X_test, y_test)



"""
gbr.save_model("output/gbr.pkl")
xtra.save_model("output/xtra.pkl")
mlpr.save_model("output/mlpr.pkl")
rft.save_model("output/rft.pkl")
cbr.save_model("output/cbr.pkl")
xgb.save_model("output/xgb.pkl")
"""


print("  ")
print(f"GBR  {gbr_score}       {gbr_mse}       {gbr_rmse}            {gbr_perf}")
print(f"XT   {xtra_score}      {xtra_mse}      {xtra_rmse}           {xtra_perf}")
print(f"MLPR {mlpr_score}      {mlpr_mse}      {mlpr_rmse}           {mlpr_perf}")
print(f"RFT  {rft_score}       {rft_mse}       {rft_rmse}            {rft_perf}")
print(f"XGB {xgb_score}      {xgb_mse}      {xgb_rmse}           {xgb_perf}")
print(f"CBR  {cbr_score}       {cbr_mse}       {cbr_rmse}            {cbr_perf}")
print("  ")



predict_list = [ gbr_predictions , xtra_predictions , mlpr_predictions , rft_predictions, cbr_predictions , xgb_predictions  ]
score_list = [ gbr_score , xtra_score , mlpr_score , rft_score, cbr_score , xgb_score ]
perf_list = [ gbr_perf , xtra_perf , mlpr_perf , rft_perf, cbr_perf , xgb_perf ]




# Combine predictions with equal weights (soft voting)
base_predictions = gbr_predictions + xtra_predictions + mlpr_predictions + rft_predictions 
all_predictions = base_predictions +  cbr_predictions + xgb_predictions


# Assign weights based on accuracy
#weights = [gbr_perf, xtra_perf, mlpr_perf, rft_perf, cbr_predictions, xgb_predictions]  # Adjust these weights based on the performance of each model
weights = [gbr_score, xtra_score, mlpr_score, rft_score, cbr_score, xgb_score]  # Adjust these weights based on the performance of each model
#weights = [0.7, 0.65, 0.72, 0.74]  # Adjust these weights based on the performance of each model

# Combine predictions using weighted average
combined_prob = (gbr_predictions * weights[0] +
                 xtra_predictions * weights[1] +
                 mlpr_predictions * weights[2] +
                 rft_predictions * weights[3] +
                 cbr_predictions * weights[4] +                
                 xgb_predictions * weights[5]                                 
                 ) / sum(weights)


correctX = 0
perfX = 0
correctY = 0
perfY = 0
correctZ = 0
perfZ = 0
totalX = 0
correctP = 0
perfP = 0
totalP = 0

for i in range(len(y_test)):
        target_output = y_test[i]  # Actual target output for the i-th sample
        prob_x = combined_prob[i]
        combined_x = all_predictions[i]        

        predict_total = (gbr_predictions[i] + xtra_predictions[i] + 
           mlpr_predictions[i] + rft_predictions[i])
        
        #print(f"Target  {target_output}    Prob  {prob_x}    Predict  {predict_total}     {predict_total_x}    ")

        
        if(target_output > 0 and (predict_total) > 0  ):
                correctX= correctX + 1 
        
        if(target_output < 0 and (predict_total) < 0 ):
                correctX= correctX + 1  
        
        if(target_output == 0 and (predict_total) == 0 ):
                correctX= correctX + 1 
                

        if(target_output > 0 and (predict_total) > 0.5  ):
                correctY= correctY + 1 
        
        if(target_output < 0 and (predict_total) < -0.5  ):
                correctY= correctY + 1         
        
        if(target_output == 0 and (predict_total) != 0  ):
                correctY= correctY + 1  


        #'------------ '

        if(target_output > 0 and combined_x> 0 ):
                correctZ= correctZ + 1 
        
        if(target_output < 0 and (combined_x) < 0 ):
                correctZ= correctZ + 1         
        
        if(target_output == 0 and (combined_x) != 0 ):
                correctZ= correctZ + 1  



        if(target_output > 0 and (combined_x> 0 or (prob_x * combined_x)  > 0) ):
                correctP= correctP + 1 
        
        if(target_output < 0 and (combined_x < 0 or (prob_x * combined_x) < 0) ):
                correctP= correctP + 1         
        
        if(target_output == 0 and (combined_x == 0 or (prob_x * combined_x) == 0) ):
                correctP= correctP + 1  

        totalX = totalX + 1    
        
        


C1 = correctX/totalX
C2 = correctY/totalX
C3 = correctZ/totalX
C4 = correctP/totalX


print(" ")
print(f"Correct X% : {correctX}     {C1}" )
print(f"Correct Y% : {correctY}     {C2}" )
print(f"Correct Z% : {correctZ}     {C3}" )
print(f"Correct P% : {correctP}     {C4}" )
print(f"Total Predicted {totalX} ")  
print(" ")
gbr.show_data_shapes()
print(" ")