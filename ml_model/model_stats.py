from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import confusion_matrix

def calc_mse_rmse_mae( y_test, predicted_values):
    
    rmse = root_mean_squared_error(y_test, predicted_values)
    mse = rmse **2.0
    mae = float(mean_absolute_error(y_test,predicted_values))          
    mae = f"{round(mae, 4):.4f}"
    mse = f"{round(mse, 4):.4f}"
    rmse = f"{round(rmse, 4):.4f}"
    return mse, rmse, mae


def gen_class_stats( y_test, predicted_values):
        
    sig_predicted_values = predicted_values        
    sig_predicted_values[sig_predicted_values > 0.5] = 1
    sig_predicted_values[sig_predicted_values < 0.5] = 0
    
    tn, fp, fn, tp = confusion_matrix(y_test, sig_predicted_values).ravel()
    mse, rmse, mae =  calc_mse_rmse_mae( y_test, sig_predicted_values)
    r2 = r2_score(y_test, sig_predicted_values)
    
    correct1 = 0 
    total = 0
    for i in range(len(y_test)):
        target_output = y_test[i] if i < len(y_test) else 0
        predicted_output = predicted_values[i]  # Predicted output for the i-th sample

        if ( target_output > 0.5 and predicted_output > 0.5):
            correct1= correct1 + 1 

        if ( target_output < 0.5 and predicted_output < 0.5):
            correct1= correct1 + 1 

        total = total + 1    

    perf = round((correct1)/total,4)
    return perf, correct1, total, tn, fp, fn, tp, mse, rmse, mae, r2


def gen_reg_stats_x( y_test, predicted_values):

    y_pred = predicted_values

    wins = 0
    losses = 0
    total = 0

    for i in range(len(y_test)):
        if (y_pred[i] > 0 and y_test[i] > 0) or (y_pred[i] < 0 and y_test[i] < 0):
            wins += 1
        elif (y_pred[i] > 0 and y_test[i] < 0) or (y_pred[i] < 0 and y_test[i] > 0):
            losses += 1
        elif (y_pred[i] == 0 and y_test[i] == 0):
            wins += 1
        elif (y_pred[i] == 0 and y_test[i] != 0):
            losses += 1
        elif (y_pred[i] != 0 and y_test[i] == 0):
            losses += 1
        else:
            print(f"{y_test[i]}   {y_pred[i]} ")            
    
    total = wins + losses
    mse, rmse, mae =  calc_mse_rmse_mae( y_test, predicted_values)
    r2 = r2_score(y_test, predicted_values)
    perf = round((wins)/total,4)
    return wins, perf, total, mse, rmse, mae, r2


def gen_reg_stats( y_test, predicted_values):
    
    correct1, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x( y_test, predicted_values)
    return perf, total, mse, rmse, mae


def calc_ensemble_results(all_predictions, estimator_run_ids):

        predictions = []
        y_target = []
        
        cxp = 0
        cyp = 0
        cpp = 0
        
        agg_rtn = 0
        agg_w_rtn = 0
        comp_rtn = 0
        agg_agree = 0
        y_count = 0        
        
        for index, row in all_predictions.iterrows():
        
            agg_predict = 0
            agg_weighted_predict = 0
        
            for id in estimator_run_ids:
        
                # predict * perf
                model_data_p = f"{id}_p"
                model_data_x_perf = row[model_data_p]
                
                # model type
                model_type_i = f"{id}_t"
                model_type = row[model_type_i]

                #untouched predict
                raw_predict = row[id]

                #Rescale for classifier
                if "Classifier" in model_type:
                    raw_predict = (row[id] - 0.5) * 2

                agg_weighted_predict += model_data_x_perf 
                agg_predict += raw_predict
                

            target_output = row['target']
            y_count_o, agg_rtn_o, agg_w_rtn_o, comp_rtn_o, agg_agree_o, comp_predict_o   = calc_ensemble(len(estimator_run_ids),  target_output, agg_predict, agg_weighted_predict )
            
            comp_predict = comp_predict_o

            if y_count_o > 0:
                y_count += y_count_o                
                agg_agree += agg_agree_o
                agg_rtn += agg_rtn_o
                agg_w_rtn += agg_w_rtn_o
                comp_rtn += comp_rtn_o
            
            y_target.append(target_output/len(all_predictions))
            predictions.append(comp_predict/len(all_predictions))

        cxp = agg_rtn/y_count
        cyp = agg_w_rtn/y_count
        cpp = comp_rtn/y_count
        
        return agg_rtn, agg_w_rtn, comp_rtn, y_count, cxp, cyp, cpp, predictions, y_target
   
   
   
def calc_ensemble(num_models, target_val, agg_pre,  agg_weighted ):
    
    agg_rtn = 0
    agg_w_rtn = 0
    comp_rtn = 0
    agg_agree = 0
    y_count = 0   
    
    agg_pre = agg_pre/num_models
    agg_predict_w = agg_weighted/num_models
    
    comp_predict = ((0.46 * agg_pre) + (0.54 * agg_predict_w)  )

    if target_val > 0:
        y_count += 1
        if agg_pre > 0 :  agg_rtn += 1
        if agg_predict_w > 0: agg_w_rtn += 1
        if comp_predict > 0: comp_rtn += 1
        if agg_weighted > 0  and agg_pre > 0:
            agg_agree += 1
        
    if target_val < 0:
        y_count += 1
        if agg_pre < 0:  agg_rtn += 1
        if agg_predict_w < 0: agg_w_rtn += 1
        if comp_predict < 0: comp_rtn += 1
        if agg_weighted < 0  and agg_pre < 0:
            agg_agree += 1                                    
            

    return y_count, agg_rtn, agg_w_rtn, comp_rtn, agg_agree, comp_predict            