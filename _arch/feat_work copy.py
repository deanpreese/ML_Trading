import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance


#file_loaded = pd.read_csv('data/sm13_3070.csv')
#file_loaded = pd.read_csv('data/buildSeqInd_Lucky13_5M_ALL.csv')
file_loaded = pd.read_csv("data/IND_LSTM_ALL.csv")
fl = file_loaded.drop(columns=['outputC'])

num_columns = len(fl.axes[1]) 
input_features =  num_columns -1
X = fl.iloc[:, 0:input_features]  
y = fl['output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {
    "n_estimators": 25,
    "max_depth": 25,
    "min_samples_split": 7,
    "learning_rate": 0.01,
    "loss": "squared_error",
    "verbose" : 1
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

print(" ")
print(params)
print(" ")
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print(" ")

print("Feature Importances")


feature_importance = reg.feature_importances_
feature_importancex = list(zip(list(fl.columns), feature_importance))
sorted_feature_importance = sorted(feature_importancex, key=lambda x: x[1], reverse=True)
for feature, weight in sorted_feature_importance:
    print(f"{feature} {weight}")

print(" ")


"""
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(list(fl.columns))[sorted_idx])
plt.title("Feature Importance (MDI)")
"""

print("Permutation Importance")
result = permutation_importance(
    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)

for ix in result.importances_mean.argsort()[::-1]:
    print(f"{list(fl.columns)[ix]} ")

print(" ")

    
"""
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(list(fl.columns))[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()
"""
