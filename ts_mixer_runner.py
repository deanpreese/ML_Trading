import pandas as pd
from models.ts_mixer_model import TSMixerModel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


datafile = [ 
    'data/Lucky13_3070_oos.csv',   
    'data/Lucky13_3070.csv',  #1
    'data/ndata_diff_lucky13_3070_oos.csv', 
    'data/ndata_diff_lucky13_3070.csv', #3
    'data/ndata_lucky_13_lag_3070_oos.csv', 
    'data/ndata_lucky13_lag_3070.csv', #5
    'new_model_Z_lucky13_3070_oos.csv',
    'new_model_Z_lucky13_3070.csv', #7,
    'data/Lucky13_3070_oos_3.csv',   
    'data/Lucky13_3070_3.csv',  #8
    'data/Lucky13_3070_oos_5.csv',   
    'data/Lucky13_3070_5.csv',  #10
]

file_path = datafile[0]

model = TSMixerModel(epochs=100, batch_size=32)
#history, y_pred = model.train_model(file_path)
#model.evaluate_model(y_pred)

model.load_saved_model()
#model.run_batch_test(file_path)
 
df = pd.read_csv(file_path)
df = df.drop(columns=['outputC'])
X = df.drop(columns=['output'])
y = df['output'].values

model.X_test = X
model.y_test = y

yn = False
count = 0
ycount = 0

y_pred = []

for i in range(len(y)):
    
    x_val = X.iloc[i]
    X_scaled = model.saved_scaler.transform([x_val])
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]
    y_val = model.model.predict(X_scaled)
    y_pred.append(y_val[0][0])

model.evaluate_model(y_pred)
    