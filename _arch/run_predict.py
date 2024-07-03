import numpy as np
import pickle
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

#from models.WrappedModels import *


# Load the trained model (replace with your model loading code)
model_filename = "output/XGBR.pkl"
loaded_model = pickle.load(open(model_filename, "rb"))
scaler_filename = "output/XGB_Scaler.pkl"        
loaded_scaler = pickle.load(open(scaler_filename, "rb"))
        
fto ="data/OOS_13.csv"
arrays_list = []

        
data = pd.read_csv(fto)           
num_columns = len(data.axes[1]) 
input_features =  num_columns -1
X = data.iloc[:, 0:input_features]  
y = data["output"].values

sum_predict = 0
sum_target = 0

predicts = []
modlist = []
tarlist = []

wronglist= []
rightlist = []

correct = 0
total = 0

current_streak = 0
last_predict = 0
win_streak_list = []
loss_streak_list = []
longest_win_streak = 0
longest_loss_streak = 0
diff_wins = []
diff_losses = []
diff_up = []
diff_down = []
tracking = []

for i in range(len(y)):

    #if i == 10:
    #    break

    d = X.iloc[i].to_numpy().reshape(1,-1)

    pred_test = loaded_scaler.transform(d)    
    loaded_prediction = loaded_model.predict(pred_test)[0]
    predicts.append(sum_target + loaded_prediction)
        
    rl = None 
    wl = None

    if  loaded_prediction > 0:
        
        diff_up.append(loaded_prediction-y[i])
        
        if y[i] > 0 :    
            
            tracking.append("W") 
            
            diff_wins.append(loaded_prediction-y[i])
            
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
        
        diff_down.append(loaded_prediction-y[i])
        
        if y[i] < 0:    
            
            tracking.append("W")
            diff_wins.append(loaded_prediction-y[i])
            diff_down.append(loaded_prediction-y[i])

                
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

       
    if rl == None:
        wl = sum_target + loaded_prediction
        tracking.append("L")
        diff_losses.append(loaded_prediction-y[i])    
       
        if current_streak > 0:
            if current_streak > longest_win_streak:
                longest_win_streak = current_streak 
                win_streak_list.append(current_streak)  
                          
            current_streak = -1 

        elif current_streak < 0:
                current_streak = current_streak - 1

        elif current_streak == 0:
               current_streak = -1       
       

    rightlist.append(rl)    
    wronglist.append(wl)
    
    sum_target = sum_target + y[i]
    tarlist.append(sum_target)
    
       

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
print(f"Ave Winning Streak {round(np.average(win_streak_list),0)}")
print(f"Ave Losing Streak {round(np.average(loss_streak_list),0)}")
print(" ")    
print(" ---------------- ")
print("Reading the data")
print(" ---------------- ")
print("If actual is 10  and Predict is 12 then predict is 2 Over " )
print("If actual is 10  and Predict is 8 then predict is -2 Under")
print(" ")    
print(f"Ave Win Miss  {round(np.average(diff_wins),4)}")
print(f"Ave Loss Miss {round(np.average(diff_losses),4)}")
print(" ")    
print(f"Ave Up Miss  {round(np.average(diff_up),4)}")
print(f"Ave Down Miss {round(np.average(diff_down),4)}")
print(" ")    

showfig = False
if showfig:
    plt.figure(figsize=(30,10), dpi=80)    
    plt.plot(predicts, 'b-')
    plt.plot(wronglist, 'rv')
    plt.plot(rightlist, 'g^')
    plt.plot(tarlist, 'k-')
    plt.show()
