from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def gen_class_stats( y_test, predicted_values):
    pass


def gen_reg_stats( y_test, predicted_values):
    
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
    return(per1, total)



def calc_class_ens_results(all_predictions, estimator_run_ids):
    pass


def calc_reg_ens_results(all_predictions, estimator_run_ids):

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
   
