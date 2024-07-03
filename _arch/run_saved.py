
import pandas as pd
from services.ModelLoader import ModelLoader


file = "data/lucky13_oos.csv"
data = pd.read_csv(file)   
num_columns = len(data.axes[1]) 
input_features =  num_columns -2
X = data.iloc[:, 0:input_features]  
y = data["output"].values

experiment_id = ['196946517036050480']
runner = ModelLoader()
models = runner.load_random_models(experiment_id, 5)
  
model_outputs = []

for m in range(len(models)):

    correct =  0
    total = 0
    
    m_filter = models[m].filter
    run_i = models[m].run_id    
    
    for i in range(len(y)):
        
        loaded_prediction = models[m].do_predict(X.iloc[i])
        
        if loaded_prediction > 0 and y[i] > 0 :
            correct += 1

        if loaded_prediction < 0 and y[i] < 0 :
            correct += 1
                
        total += 1        
        
    model_outputs.append([correct,total, (correct/total), len(m_filter), run_i])            
  
            
m_df = pd.DataFrame(model_outputs)
m_df.columns = ["Correct", "Total", "%" ,  "Filter Count" ,  "Run Id"]            
print(m_df)
