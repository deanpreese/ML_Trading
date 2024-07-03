from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_param_list(grid):
    param_combinations = list(product(*grid.values()))
    par_list = []
    for params in param_combinations:
        param_set = dict(zip(grid.keys(), params))
        par_list.append(param_set)
    
    return par_list


def simple_split_and_scale(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.columns = X_train.columns.str.replace('[^+a-zA-Z0-9]', '_')
    X_test.columns = X_test.columns.str.replace('[^+-a-zA-Z0-9]', '_')
    
    return X_train, X_test, y_train, y_test


def full_split_and_scale_with_filter(pd_data, col_offset, size_test, random_state, output_col, filter):
    num_columns = len(pd_data.axes[1])  
    input_features =  num_columns - col_offset
    X = pd_data.iloc[:, 0:input_features]  
    X = X.iloc[:, filter]  
    y = pd_data[output_col].values
    X_train, X_test, y_train, y_test,  =  simple_split_and_scale(X, y, size_test, random_state)
    
    return X_train, X_test, y_train, y_test, input_features, X.columns


def full_split_and_scale(pd_data, col_offset, size_test, random_state, output_col):
    num_columns = len(pd_data.axes[1]) 
    input_features =  num_columns - col_offset
    X = pd_data.iloc[:, 0:input_features]  
    y = pd_data[output_col].values
    X_train, X_test, y_train, y_test  =  simple_split_and_scale(X, y, size_test, random_state)
    
    return X_train, X_test, y_train, y_test, input_features


def write_line_to_file(file, txt):
    file1 = open(file, "a")  # append mode
    file1.write(txt + "\n" )
    file1.close()


def calc_mix_max(datax):
    confidence_level = 0.8
    mm = []
    column_parameters = []
    for column in datax.columns:
        mean = datax[column].mean()
        std_dev = datax[column].std()
        column_parameters.append((mean, std_dev))

    z_score = norm.ppf((1 + confidence_level) / 2)
    min_max_values = []
    for (mean, std_dev) in column_parameters:
        min_value = mean - z_score * (std_dev* 0.75)
        max_value = mean + z_score * (std_dev* 0.75)
        min_max_values.append((min_value, max_value))

    for i, column in enumerate(datax.columns):
        #print(f'Column "{column}": Normal Min={min_max_values[i][0]}, Normal Max={min_max_values[i][1]}')
        t = [min_max_values[i][0], min_max_values[i][1]]
        mm.append(t)

    return mm


def gen_importances(features, importances, display):
    features_list = []
    feature_importance = list(zip(features, importances))
    sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    for feature, weight in sorted_feature_importance:
        txt = f"{feature} {weight}"
        features_list.append(txt)
        
    return features_list


def show_stats( DisplayOutput, y_test, predicted_values):
    correct1 = 0 
    total = 0
    for i in range(len(y_test)):
        target_output = y_test[i] if i < len(y_test) else 0
        predicted_output = predicted_values[i]  # Predicted output for the i-th sample

        if ( target_output > 0 and predicted_output > 0):
            correct1= correct1 + 1 

        if ( target_output < 0 and predicted_output < 0):
            correct1= correct1 + 1 
        
        if ( target_output == 0 and predicted_output == 0):
            correct1= correct1 + 1     

        total = total + 1    

    per1 = round((correct1)/total,4)

    if DisplayOutput:
        print(f"Correct% : {correct1}     {per1}")
        print(f"Total Predicted {total}")
        
    return(per1, total)


def calc_MSE(y_test, predicted_values, display):
    mse = mean_squared_error(y_test, predicted_values)
    rmse = mse**.5
    
    if display:
        print(f"MSE {mse}")
        print(f"RMSE {rmse}")
    
    return (mse, rmse)


def calc_class_results_X(all_predictions, models):
    correctX = 0
    correctY = 0
    totalX = 0
    correctP = 0
        
    for index, row in all_predictions.iterrows():
        key_result = 0
        weighted_prob = 0
    
        for key in models.keys():
            key_result += row[key]
            
            k = f"{key}-proba"
            prob_data = row[k][0]
            weighted_prob = key_result * prob_data
            
        target_output = row['target']
        prob_z = weighted_prob/len(models)
            
        #print(f"{key_result}  {row['target']} {prob_z}")
                
        if(target_output > 0 and (key_result) > 0  ):
                correctX= correctX + 1 
        
        if(target_output == 0 and (key_result) == 0 ):
                correctX= correctX + 1 
                

        if(target_output > 0 and ((prob_z > 0.5))  ):
                correctY= correctY + 1 
        
        if(target_output == 0 and ((prob_z < 0.5))  ):
                correctY= correctY + 1  


        if(target_output > 0 and ((key_result > 0)  or (prob_z > 0.5))  ):
                correctP= correctP + 1 
        
        if(target_output == 0 and ((key_result == 0)  or (prob_z < 0.5))  ):
                correctP= correctP + 1  

        totalX = totalX + 1    
                
            
    cxp = correctX/totalX
    cyp = correctY/totalX
    cpp = correctP/totalX

    return correctX, correctY, correctP, totalX, cxp, cyp, cpp

def calc_class_results(all_predictions, estimator_run_ids):

        r_predictions = []
        r_y_target = []
        
        correctX = 0
        correctY = 0
        totalX = 0
        correctP = 0
        cxp = 0
        cyp = 0
        cpp = 0
        
        for index, row in all_predictions.iterrows():
        
            agg_predict = 0
            agg_weighted_predict = 0
        
            # for each id pull the result and prob data 
            for id in estimator_run_ids:
                agg_predict += row[id]
        
                k = f"{id}-proba"
                prob_data = row[k][0]
                agg_weighted_predict = agg_predict * prob_data
        
               
            target_output = row['target']
            prob_z = agg_weighted_predict/len(estimator_run_ids)
            
            wp_kr = (agg_weighted_predict * agg_predict) 
            
            if(target_output > 0 and (agg_predict) > 0  ):
                    correctX= correctX + 1 
            
            if(target_output == 0 and (agg_predict) == 0 ):
                    correctX= correctX + 1 
                    

            if(target_output > 0 and ((prob_z > 0.5))  ):
                    correctY= correctY + 1 
            
            if(target_output == 0 and ((prob_z < 0.5))  ):
                    correctY= correctY + 1  


            if(target_output > 0 and ((agg_predict > 0)  or (prob_z > 0.5))  ):
                    correctP= correctP + 1 
            
            if(target_output == 0 and ((agg_predict == 0)  or (prob_z < 0.5))  ):
                    correctP= correctP + 1  

            totalX = totalX + 1    
            
            local_predict = 0
            if agg_predict > 0 or agg_weighted_predict > 0:
                local_predict = 1
            elif agg_predict < 0 or agg_weighted_predict < 0:
                local_predict = -1
            else:
                local_predict = 0                
            

            r_y_target.append(target_output)
            r_predictions.append(local_predict)
        
        cxp = correctX/totalX
        cyp = correctY/totalX
        cpp = correctP/totalX
        
        return correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target


def calc_reg_results(all_predictions, estimator_run_ids):

        r_predictions = []
        r_y_target = []
        
        correctX = 0
        correctY = 0
        totalX = 0
        correctP = 0
        cxp = 0
        cyp = 0
        cpp = 0
        
        for index, row in all_predictions.iterrows():
        
            agg_predict = 0
            agg_weighted_predict = 0
        
            # for each id pull the result and prob data 
            for id in estimator_run_ids:
                agg_predict += row[id]
        
                k = f"{id}_p"
                prob_data = row[k]
                agg_weighted_predict = agg_predict * prob_data
                
            target_output = row['target']
            
            wp_kr = (agg_weighted_predict * agg_predict) 
            
                        
            if(target_output > 0 and (agg_predict) > 0  ):
                    correctX= correctX + 1 
            
            if(target_output < 0 and (agg_predict) < 0 ):
                    correctX= correctX + 1  
            
            if(target_output == 0 and (agg_predict) == 0 ):
                    correctX= correctX + 1 
                    

            if(target_output > 0 and (wp_kr) > 0  ):
                    correctY= correctY + 1 
            
            if(target_output < 0 and (wp_kr) < 0  ):
                    correctY= correctY + 1         
            
            if(target_output == 0 and (wp_kr) == 0  ):
                    correctY= correctY + 1  


            if(target_output > 0 and (agg_predict> 0 or (wp_kr)  > 0) ):
                    correctP= correctP + 1 
                    
            if(target_output < 0 and (agg_predict < 0 or (wp_kr) < 0) ):
                    correctP= correctP + 1         
            
            if(target_output == 0 and (agg_predict == 0 or (wp_kr) == 0) ):
                    correctP= correctP + 1  

            totalX = totalX + 1 
            
            local_predict = 0
            if agg_predict > 0 or agg_weighted_predict > 0:
                local_predict = 1
            elif agg_predict < 0 or agg_weighted_predict < 0:
                local_predict = -1
            else:
                local_predict = 0

            r_y_target.append(target_output)
            r_predictions.append(local_predict)
        
        cxp = correctX/totalX
        cyp = correctY/totalX
        cpp = correctP/totalX
        
        return correctX, correctY, correctP, totalX, cxp, cyp, cpp, r_predictions, r_y_target
   

def calc_reg_streaks(predictions, y_test, display_data, display_plot, asList):

    sum_target = 0
    predicts = []
    tar_list = []

    wrong_list= []
    right_list = []

    correct = 0

    current_streak = 0
    win_streak_list = []
    loss_streak_list = []
    longest_win_streak = 0
    longest_loss_streak = 0
    diff_wins = []
    diff_losses = []
    diff_up = []
    diff_down = []
    tracking = []


    for i in range(len(y_test)):

        loaded_prediction = predictions[i]
        predicts.append(sum_target + loaded_prediction)
            
        rl = None 
        wl = None

        if  loaded_prediction > 0:
            
            diff_up.append(loaded_prediction-y_test[i])
            
            if y_test[i] > 0 :    
                
                tracking.append("W") 
                
                diff_wins.append(loaded_prediction-y_test[i])
                
                rl = sum_target + loaded_prediction
                correct += 1
            
                if current_streak < 0:
                    if current_streak < longest_loss_streak:
                        longest_loss_streak = current_streak
                        loss_streak_list.append(current_streak) 
                        
                    current_streak = 1         

                elif current_streak > 0:
                    current_streak = current_streak + 1

                elif current_streak == 0:
                    current_streak = 1       
        
        if  loaded_prediction < 0:
            
            diff_down.append(loaded_prediction-y_test[i])
            
            if y_test[i] < 0:    
                
                tracking.append("W")
                diff_wins.append(loaded_prediction-y_test[i])
                diff_down.append(loaded_prediction-y_test[i])
                    
                rl = sum_target + loaded_prediction
                correct += 1
                
                if current_streak < 0:
                    if current_streak < longest_loss_streak:
                        loss_streak_list.append(current_streak)  
                        
                    current_streak = 1         

                elif current_streak > 0:
                    current_streak = current_streak + 1

                elif current_streak == 0:
                    current_streak = 1             
        
        if rl is None:
            
            wl = sum_target + loaded_prediction
            tracking.append("L")
            diff_losses.append(loaded_prediction-y_test[i])    
        
            if current_streak > 0:
                if current_streak > longest_win_streak:
                    longest_win_streak = current_streak 
                    win_streak_list.append(current_streak)  
                            
                current_streak = -1 

            elif current_streak < 0:
                    current_streak = current_streak - 1

            elif current_streak == 0:
                current_streak = -1       
        

        right_list.append(rl)    
        wrong_list.append(wl)
        
        sum_target = sum_target + y_test[i]
        tar_list.append(sum_target)
        
    
    c_perc = round(correct/i,4)
    aws = round(np.average(win_streak_list),0)
    als = round(np.average(loss_streak_list),0)
    awm = round(np.average(diff_wins),4)
    alm = round(np.average(diff_losses),4)
    aum = round(np.average(diff_up),4)
    adm = round(np.average(diff_down),4)
    
    out_txt = [i, correct, c_perc , current_streak, longest_win_streak, longest_loss_streak, aws, als, awm, alm, aum, adm ]    
       
    if display_data:
        print(" ---------------- ")    
        print(" ")        
        print(f"Total Predicts {i}") 
        print(f"Total Correct  {correct}")
        print(f"Correct Percentage {round(correct/i,4)}")   
        print(" ")    
        print(f"Current Streak {current_streak}")
        print(f"Longest Winning Streak {longest_win_streak}")
        print(f"Longest Losing Streak {longest_loss_streak}")
        print(" ")    
        print(f"Ave Winning Streak {aws}")
        print(f"Ave Losing Streak {als}")
        print(" ")    
        print(" ---------------- ")
        print("Reading the data")
        print(" ---------------- ")
        print("If actual is 10  and Predict is 12 then predict is 2 Over " )
        print("If actual is 10  and Predict is 8 then predict is -2 Under")
        print(" ")    
        print(f"Ave Win Miss  {awm}")
        print(f"Ave Loss Miss {alm}")
        print(" ")    
        print(f"Ave Up Miss  {aum}")
        print(f"Ave Down Miss {adm}")
        print(" ")    

    if display_plot:
        plt.figure(figsize=(30,10), dpi=80)    
        plt.plot(predicts, 'b-')
        plt.plot(wrong_list, 'rv')
        plt.plot(right_list, 'g^')
        plt.plot(tar_list, 'k-')
        plt.show()

    if asList:
        return out_txt
    else:
        return current_streak, longest_win_streak, longest_loss_streak, aws, als, awm, alm, aum, adm