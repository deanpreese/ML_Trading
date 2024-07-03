import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from common_func import create_param_list
from wrapped_models import ModelType, gen_classifier_data, gen_regressor_data

# -----------------------------------------------------
#
#   Classifier Wrappers   
#
# -----------------------------------------------------
class TunableBaggingClassifier(BaggingClassifier):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)
 
    def model_type(self):
        return ModelType.CLASS
 
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(model, "TunableBaggingClassifier", signature=signature)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba  
      

# -----------------------------------------------------
class TunableLogisticRegression(LogisticRegression):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS

    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableLogisticRegression", signature=signature)
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
            
    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba             
        
# -----------------------------------------------------
class TunableAdaBoostClassifier(AdaBoostClassifier):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS

    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableAdaBoostClassifier", signature=signature)
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba 
    

# -----------------------------------------------------
class TunableGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableGradientBoostingClassifier", signature=signature)
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba 


# -----------------------------------------------------
class TunableRandomForestClassifier(RandomForestClassifier):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS

    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableRandomForestClassifier", signature=signature)
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)
    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba 


# -----------------------------------------------------
class TunableGaussianNB(GaussianNB):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.CLASS
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableGaussianNB", signature=signature)
            accuracy, precision, recall, TN, FP, FN, TP, tot = gen_classifier_data(model, X_train, y_train, X_test, y_test, y_pred)

    
        return accuracy, precision, recall, TN, FP, FN, TP, tot, y_pred, pred_proba 
    
    
# -----------------------------------------------------    
# -----------------------------------------------------        
# -----------------------------------------------------
#
#   Regressor Wrappers   
#
# -----------------------------------------------------        
# -----------------------------------------------------
# -----------------------------------------------------

# -----------------------------------------------------
class TunableBaggingRegressor(BaggingRegressor):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.REGR
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableBaggingRegressor", signature=signature)
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred

# -----------------------------------------------------
class TunableAdaBoostRegressor(AdaBoostRegressor):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.REGR
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableAdaBoostRegressor", signature=signature)
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred

# -----------------------------------------------------
class TunableGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog()
        self.features_used = []
        self.used_params = kwargs
        super().__init__(**kwargs)

    def model_type(self):
        return ModelType.REGR
    
    def param_grid(self):
    
        grid = {
            'max_depth': [25, 35, 45, 50, 100, 150],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 2,3,4, 5, 10, 15],
            'min_samples_split': [1,2,3,4,5,10, 15],
            'n_estimators': [75, 80,90,100, 125, 150, 200, 500, 1000, 1500, 2000]
        }    
    
        return grid
    
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableGradientBoostingRegressor", signature=signature)
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred
    
    
        
# -----------------------------------------------------
class TunableRandomForestRegressor(RandomForestRegressor):
    def __init__(self, **kwargs):
        mlflow.sklearn.autolog(exclusive=False)        
        self.used_params = kwargs
        self.features_used = []
        super().__init__(**kwargs)
        
    def model_type(self):
        return ModelType.REGR
    
    def param_grid(self):
        
        rf_grid = {
            'max_depth': [75, 100, 150],
            'min_samples_leaf': [1,  3, 5, 10, 15],
            'min_samples_split': [1,  3, 5, 10, 15],
            'n_estimators': [100],
            'verbose': [1],
            'n_jobs' : [-1],
            }  
        
        
        return create_param_list(rf_grid)
        
    def track_model(self, experiment_id, nested, model, X_train, y_train, X_test, y_test):
        with mlflow.start_run(experiment_id = experiment_id, nested=nested):
            self.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            signature = infer_signature(X_train, y_pred)
            mlflow.log_params( self.used_params )
            mlflow.log_table(data=pd.DataFrame(self.features_used), artifact_file="features_used.json")            
            mlflow.sklearn.log_model(model, "TunableRandomForestRegressor", signature=signature)
            perf, tot, mse, rmse, r2, score, mae = gen_regressor_data(model, X_train, y_train, X_test, y_test, y_pred)
        
        return perf, tot, mse, rmse, r2, score, mae, y_pred