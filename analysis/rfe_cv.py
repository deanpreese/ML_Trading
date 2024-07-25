from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

datafile = [ 
        '../data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        '../data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        '../data/buildSeqInd_Lucky13_F.csv',  #2
        '../data/buildSeqInd_Lucky13_D.csv',  #3
        '../data/buildSeqInd_Lucky13_F_3070.csv',  #4
        '../data/Expanded_Lucky13_3070.csv',  #5
        '../data/alt_ex13.csv', #6
        
        
    ]

file_loaded = pd.read_csv(datafile[6])
X = file_loaded
X = X.drop(columns=['output', 'outputC'])
y = file_loaded['outputC'].values
feature_names = list(X.columns)

min_features_to_select = 3  # Minimum number of features to consider
clf = RandomForestClassifier()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=-1,
    verbose=2
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")


cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()