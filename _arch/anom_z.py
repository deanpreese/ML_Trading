import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import IncrementalPCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_model.model_stats import gen_reg_stats

class RealTimeAnomalyDetection:
    def __init__(self):
        self.anomaly_models = []
        self.anomaly_flags = None
        self.anomaly_stats = {}

    def build_and_train_anomaly_ensemble(self, X_train):
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso_x = IsolationForest(random_state=42)
        ipca = IncrementalPCA(n_components=2)
        self.anomaly_models = [iso_forest, ipca, iso_x]
        
        for model in self.anomaly_models:
            if isinstance(model, IsolationForest):
                model.fit(X_train)
            elif isinstance(model, IncrementalPCA):
                model.fit(X_train)

    def flag_anomalies(self, X_test):
        anomaly_scores = np.zeros(len(X_test))
        model_anomaly_counts =  {}

        for model in self.anomaly_models:
            if isinstance(model, IsolationForest):
                scores = model.decision_function(X_test)
                anomaly_scores += scores
                model_anomaly_counts['IsolationForest'] = np.sum(scores < np.percentile(scores, 5))
                print(scores)
                
            elif isinstance(model, IncrementalPCA):
                reconstruction_error = np.sum((X_test - model.inverse_transform(model.transform(X_test)))**2, axis=1)
                anomaly_scores += reconstruction_error
                model_anomaly_counts['IncrementalPCA'] = np.sum(reconstruction_error > np.percentile(reconstruction_error, 95))

        self.anomaly_flags = anomaly_scores < np.percentile(anomaly_scores, 5)  # Top 5% as anomalies
        total_anomalies = np.sum(self.anomaly_flags)
        self.anomaly_stats = {
            'Total Anomalies Detected': total_anomalies,
            'Model Anomaly Counts': model_anomaly_counts
        }
        
        print(f"Total anomalies detected: {total_anomalies}")
        
        for key in model_anomaly_counts:
            print(key, "->", model_anomaly_counts[key])
        
        print(f"Anomaly counts per model: {model_anomaly_counts['IsolationForest']}   {model_anomaly_counts['IncrementalPCA']}  ")
        return self.anomaly_stats, self.anomaly_flags


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    wins = np.sum((y_pred > 0) & (y_true > 0)) + np.sum((y_pred < 0) & (y_true < 0))
    losses = np.sum((y_pred < 0) & (y_true > 0))  + np.sum((y_pred > 0) & (y_true < 0))  
    total = wins+losses
    win_percentage = (wins / total) 
   
    return mse, r2, wins, losses, win_percentage

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values
    return train_test_split(X, y, test_size=0.7, random_state=42)


def evaluate_baseline(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse, r2, wins, losses, win_percent = calculate_metrics(y_test, y_pred)
    print(f"Baseline - MSE: {mse}, R2: {r2}, Wins: {wins}, Losses: {losses}, Win Percent: {win_percent:.2f}%")


def evaluate_filtered(model, X_test, y_test, anomaly_flags):
    non_anomalous_X = X_test[~anomaly_flags]
    non_anomalous_y = y_test[~anomaly_flags]
    y_pred = model.predict(non_anomalous_X)
    mse, r2, wins, losses, win_percent = calculate_metrics(non_anomalous_y, y_pred)
    print(f"Filtered - MSE: {mse}, R2: {r2}, Wins: {wins}, Losses: {losses}, Win Percent: {win_percent:.2f}%")


def main():
    filepath='data/Lucky13_3070.csv'
    X_train, X_test, y_train, y_test = preprocess_data(filepath)
    
    detector = RealTimeAnomalyDetection()
    detector.build_and_train_anomaly_ensemble(X_train)
    anomaly_stats, anomaly_flags = detector.flag_anomalies(X_test)
    
    
    #model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    evaluate_baseline(model, X_test, y_test )
    evaluate_filtered(model, X_test, y_test, anomaly_flags)

if __name__ == "__main__":
    main()