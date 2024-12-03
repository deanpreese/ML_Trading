from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error,r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score


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
        
            class_scaled_predict = 0
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
                scaled_predict = 0

                #Rescale Regressor to -1 to 1 to match the Classifier rescale
                if "Regressor" in model_type:
                    if raw_predict > 0:
                        scaled_predict = 1
                    if raw_predict < 0:
                        scaled_predict = -1                        
                    if raw_predict == 0:
                        scaled_predict = 0

                #Rescale for classifier to -1 to 1
                if "Classifier" in model_type:
                    scaled_predict = (raw_predict - 0.5) * 2

                agg_weighted_predict += model_data_x_perf 
                class_scaled_predict += scaled_predict
                
            target_val = row['target']
            
            agg_class_pre = class_scaled_predict/len(estimator_run_ids)
            agg_predict_w = agg_weighted_predict/len(estimator_run_ids)
            comp_predict = ((0.5 * agg_class_pre) + (0.5 * agg_predict_w)  )

            y_count += 1
            
            if target_val > 0:
                if agg_class_pre > 0 :  agg_rtn += 1
                if agg_predict_w > 0: agg_w_rtn += 1
                if comp_predict > 0: comp_rtn += 1
                if agg_predict_w > 0  and agg_class_pre > 0:
                    agg_agree += 1
                
            if target_val < 0:
                if agg_class_pre < 0:  agg_rtn += 1
                if agg_predict_w < 0: agg_w_rtn += 1
                if comp_predict < 0: comp_rtn += 1
                if agg_predict_w < 0  and agg_class_pre < 0:
                    agg_agree += 1                                    
                    
            if target_val == 0:
                if agg_class_pre == 0:  agg_rtn += 1
                if agg_predict_w == 0: agg_w_rtn += 1
                if comp_predict == 0: comp_rtn += 1
                if agg_predict_w == 0  and agg_class_pre == 0:
                    agg_agree += 1                  
            
            #y_target.append(target_output/len(all_predictions))
            y_target.append(target_val)
            predictions.append(agg_class_pre)

        #scaled_predictions = [1 if x > 0 else 0 for x in predictions]
        scaled_predictions = np.where(np.array(predictions) > 0, 1, 0)
        scaled_target = np.where(np.array(y_target) > 0, 1, 0)
        
        tn, fp, fn, tp = confusion_matrix(scaled_target, scaled_predictions).ravel()
        #mse, rmse, mae =  calc_mse_rmse_mae( y_target, scaled_predictions)
        #r2 = r2_score(y_target, predictions)

        accuracy = round(accuracy_score(scaled_target, scaled_predictions),4)
        precision = round(precision_score(scaled_target, scaled_predictions),4)
        recall = round(recall_score(scaled_target, scaled_predictions),4)

        total = tn + fp + fn + tp 
        win_p = round((tn + tp)/total,4)
        loss_p = round((fp + fn)/total,4)
        
        tn_p = round(tn/total,4)
        tp_p = round(tp/total,4)
        fn_p = round(fn/total,4)
        fp_p = round(fp/total,4)
        
        #print(f" {win_p }  {loss_p}  {tn_p} {fp_p} {fn_p}  {tp_p}")

        return  accuracy, precision, recall, win_p, loss_p, tn_p, tp_p, fn_p, fp_p, predictions, scaled_predictions, y_target
   
      