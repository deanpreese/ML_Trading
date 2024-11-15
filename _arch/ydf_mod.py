import ydf
import pandas as pd

from ml_model.model_stats import gen_reg_stats_x, gen_class_stats  

datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_3070_AUG_oos.csv',   
        'data/Lucky13_3070_AUG.csv',  #3
]
df = pd.read_csv(datafile[1])
df = df.drop(columns=['ATR54','ATR53','ATR51', 'ATR52','SDKC91','SDKC9', 'ATR21','SDBB91','SDLR310'])

#df = df.drop(columns=['ActualClose','ATR53','ATR51', 'ATR52','SDKC91','SDKC9', 'LREMA921', 'TV21',
#                      'TV31','TV32', 'LREMA2150','DayOfWeek','LREMA21', 'MinOfHour', 'ATR21','SDBB91','SDLR310', 'outputC'])

#df = df.drop(columns=['outputC' ])
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = 0.8  # 80% training data
train_count = int(len(df) * train_size)
df_t = df_shuffled.iloc[:train_count]  # First 80% for training
df_tt = df_shuffled.iloc[train_count:]   # Remaining 20% for testing

df_train = df_t.drop(columns='outputC')
df_test = df_tt.drop(columns='outputC')
y_test = df_test['output'].values

df_train_c = df_t.drop(columns='output')
df_train_c['outputC'] = df_train_c['outputC'].astype(int)
df_test_c = df_tt.drop(columns='output')
y_test_c = df_test_c['outputC']

df_test_c['outputC'] = df_test_c['outputC'].astype(int)
y_test_c = y_test_c.values

# Train a Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label="output", task=ydf.Task.REGRESSION).train(df_train, verbose=0)
model_c = ydf.GradientBoostedTreesLearner(label="outputC", task=ydf.Task.CLASSIFICATION).train(df_train_c, verbose=0)

# Look at a model (input features, training logs, structure, etc.)
#print(model.describe())

# Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
#evaluation = model.evaluate(df_test)
#print(evaluation)

# Generate predictions
y_pred = model.predict(df_test)
y_pred_c = model.predict(df_test_c)

correct, perf, total, mse, rmse, mae, r2 = gen_reg_stats_x(y_test, y_pred)
print(" ")
print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
print(f"Total Wins: {correct}, Total Losses: {total-correct}, Win Percentage: {perf:.4f}")
print(f"Number of Samples: {total}")        
print(" ")        


perf, correct1, total, tn, fp, fn, tp, mse, rmse, mae, r2 = gen_class_stats(y_test_c, y_pred_c)
print(" ")
print(f"TN {tn}   FP {fp}  FN {fn}  TP {tp}")
print(f"Pred MSE: {mse},  MAE: {mae}, R2: {r2}")
print(f"Total Wins: {correct1}, Total Losses: {total-correct1}, Win Percentage: {perf:.4f}")
print(f"Number of Samples: {total}")        
print(" ")        



# Analyse a model (e.g. partial dependence plot, variable importance)
#analysis = model.analyze(df_test)
#analysis.to_file('ydf_anlysis.html')


# Benchmark the inference speed of a model
#model.benchmark(df_test)
