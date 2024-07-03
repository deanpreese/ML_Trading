import mlflow
import pandas as pd

mlflow.set_tracking_uri(uri="http://127.0.0.1:8888")
logged_model = 'runs:/9a45a4871b8443ca8134346f2b3dda89/TunableXGBRegressor'
loaded_model = mlflow.pyfunc.load_model(logged_model)

file = "data/lucky13_oos.csv"
data = pd.read_csv(file)   

num_columns = len(data.axes[1]) 
input_features =  num_columns -2
X = data.iloc[:, 0:input_features]  
y = data["output"].values

correct =  0
total = 0


for i in range(len(y)):
        
        d = X.iloc[i].to_numpy().reshape(1,-1)
        df = pd.DataFrame(d)
        
        pred = loaded_model.predict(df)
        
        print(pred)
        
        pred_1 = pred[0]
        
        if pred_1 > 0 and y[i] > 0 :
            correct += 1

        if pred_1 < 0 and y[i] < 0 :
            correct += 1

                
        total += 1        
        
        
print(f" {correct}   {total} ")