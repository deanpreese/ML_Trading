import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from ml_model.model_stats import gen_reg_stats
from ml_model.data_func import simple_split_and_scale


def calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, features):

    importance_df = pd.DataFrame(features, columns=['Feature'])
    baseline_data =[]
    
    # Train models, calculate importances, and store results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred)
            baseline_performance = perf
        else:           
            baseline_performance = accuracy_score(y_test, y_pred)
            
        baseline_data.append({'Model': name, 'Baseline': baseline_performance, 'R2': model_r2, 'MSE':mse })
        print(f"Baseline {baseline_performance}")        
        print(f"Calculating Importances ... ")        
        
        perm_importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
        perm_means = perm_importances.importances_mean
        perm_stds = perm_importances.importances_std
        
        # Calculate standard feature importance from the model
        if 'Cat' in name:
            standard_importances = model.get_feature_importance()
        else:
            standard_importances = model.feature_importances_
        
        # Store results in DataFrame
        importance_df[f'{name}_PI_Mean'] = perm_means
        importance_df[f'{name}_PI_Std'] = perm_stds
        importance_df[f'{name}_Std_Imp'] = standard_importances

    # Compute composite importance
    importance_df['Comp_PI_Mean'] = importance_df[
        [f'{name}_PI_Mean' for name in models.keys()]
    ].mean(axis=1)

    importance_df['Comp_Std_Imp'] = importance_df[
        [f'{name}_Std_Imp' for name in models.keys()]
    ].mean(axis=1)

    importance_df_sorted = importance_df.sort_values(by='Comp_Std_Imp', ascending=False)
    
    baseline_df = pd.DataFrame(baseline_data)
    print(baseline_df)
    
    return importance_df_sorted, baseline_df
    

def select_features(importance_df_sorted, threshold_v):    

    # Define thresholds for feature selection
    threshold_mean = np.percentile(importance_df_sorted['Comp_PI_Mean'], threshold_v)  # top 25% permutation importance mean
    threshold_standard = np.percentile(importance_df_sorted['Comp_Std_Imp'], threshold_v)  # top 25% standard importance

    # Select features that are consistently important
    important_features = importance_df_sorted[
        (importance_df_sorted['Comp_PI_Mean'] > threshold_mean) & 
        (importance_df_sorted['Comp_Std_Imp'] > threshold_standard)
    ]['Feature'].tolist()
    return important_features
    
def retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df):    
    
    composite_results = []
    for name, model in models.items():
        X_train_selected = X_train[important_features]
        X_test_selected = X_test[important_features]

        model.fit(X_train_selected, y_train)
        
        # Evaluate the new model
        y_pred_selected = model.predict(X_test_selected)
        sel_mse = mean_squared_error(y_test, y_pred_selected)
        sel_model_r2 = r2_score(y_test,y_pred_selected)
        
        if "Reg" in name:
            perf, total, mse, rmse, mae = gen_reg_stats(y_test, y_pred_selected)
            selected_performance = perf
        else:           
            selected_performance = accuracy_score(y_test, y_pred_selected)
        
        baseline_perf = baseline_df[baseline_df['Model'] == name]['Baseline'].values[0]
        base_r2 = baseline_df[baseline_df['Model'] == name]['R2'].values[0]
        base_mse = baseline_df[baseline_df['Model'] == name]['MSE'].values[0]
        
        composite_results.append({'Model': name, 'Perf': baseline_perf, 'Sel_Perf': selected_performance,  'R2': base_r2, 'Sel_R2': sel_model_r2, 'MSE': base_mse, 'Sel_MSE': sel_mse})
        #print(f'{name} Selected Features Performance: {selected_performance:.4f}')

    # Convert results to DataFrame and plot performance comparison
    performance_df = pd.DataFrame(composite_results) 
    return performance_df   




def gen_results(models, X_train, y_train, X_test, y_test, columns, threshold):
    
    importance_df_sorted, baseline_df = calc_importances_and_baseline(models, X_train, y_train, X_test, y_test, columns)
    important_features = select_features(importance_df_sorted, threshold)    
    performance_df = retrain_models(models, important_features, X_train, X_test, y_train, y_test, baseline_df)   
   
    return importance_df_sorted, important_features, performance_df 
    
    
# --------------
def run():

    datafile = [ 
            'data/Lucky13_3070_oos.csv',   
            'data/Lucky13_3070.csv',  #1
            'data/Model_X_3070_oos.csv',  
            'data/Model_X_3070.csv',  #3        
            'data/The_13_X_3070_oos.csv',
            'data/The_13_X_3070.csv', #5
            'data/Model_M_1_3070.csv' #6
    ] 

    df = pd.read_csv(datafile[6])

    #df = df[((df['RSIRAW'] > 50))]  
    #df = df[((df['RSIRAW'] > 70))]  
    
    
    #df = df[((df['RSIRAW'] > 0) & (df['RSIRAW'] < 30))|   
    #    ((df['RSIRAW'] > 70) & (df['RSIRAW'] < 100))]  
    
    #df = df[((df['RSI'] > 0) & (df['RSI'] < 30))]  
    #df = df[((df['RSI'] > 70) & (df['RSI'] < 100))]  
    
    model_m_1 = [
    #"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour",
    #"SeqClose", "RSIRAW", 
    "SDBB9L", "SDBB9U", "SDKC7U", "SDKC7L",
    "ROC14", "ROC9", "ATR5", "ATR2", "RSI14", "RSI14Avg", "RSIH14", "RSIL14",
    "RSI9", "RSI9Avg", "RSIH9", "RSIL9", "STOK721", "STOK513", "STOK721D", "STOK513D",
    "TV1", "TV2", "TV3", "TV4", "TV5", "TV6", "COMP0", "COMP1", "COMP2", "COMP3"
    ]

    X = df[model_m_1]
    
    
    #X = df[fd]
    #X = X.drop(columns=['output', 'outputC', 'SeqClose', 'RSIRAW',])
    
    lucky_13_columns = [
    "SDLR310", "SDBB91", "SDKC91", "SDKC9", "ROC", "ATR54", "ATR53", "ATR52", 
    "ATR51", "ATR5", "ATR21", "ATR2", "RSI", "STOK1"
    ]
    #X = df[lucky_13_columns]
    
    model_x_columns = [
        "Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour", "RSIRAW", "SeqClose",
        "SDBB9", "SDBB91", "SDKC9", "SDKC91", "SDBB29", "SDBB291", "SDKC29", "SDKC291",
        "SDBB14CU", "SDBB14CL", "SDBB9CU", "SDBB9CL", "SDKC10CU", "SDKC10CL", "SDKC7CU",
        "SDKC7CL", "ROC14", "ROC9", "ROC7", "ATR14", "ATR9", "ATR5", "ATR2", "RSI14", 
        "RSI9", "ADX14", "ADX9", "STO5135K", "STO5135D", "STO7143K", "STO7143D", "TV1", 
        "TV2", "TV3", "TV4", "TV5", "TV6", "ZH79X", "ZL79X", "ZC79X", "COMP0", "COMP1", 
        "COMP2", "COMP3"
    ]
    #X = df[model_x_columns]
    

    y = df['outputC'].values
    y2 = df['output'].values
    
    threshold = 50

    #X_train_c, X_test_c, y_train_c, y_test_c = simple_split_and_scale(X, y, 0.2, 42)
    #X_train_r, X_test_r, y_train_r, y_test_r = simple_split_and_scale(X, y2, 0.2, 42)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y2, test_size=0.2, random_state=42)
    
    models_c = {
        #'RandomForestClassifier' :RandomForestClassifier(random_state=42, verbose=2, n_jobs=-1),
        'LGBCls': LGBMClassifier(random_state=42, verbose=2, n_jobs=-1),
        'XGBCls': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=2),
        'CatCls': CatBoostClassifier(random_state=42, verbose=2)
    }

    models_r = {
        #'RandomForestRegressor' :RandomForestRegressor(random_state=42, verbose=2, n_jobs=-1),
        'LGBReg': LGBMRegressor(random_state=42, verbose=2),
        'XGBReg': XGBRegressor(random_state=42, use_label_encoder=False, verbosity=2),
        'CatReg': CatBoostRegressor(random_state=42, verbose=2)
    }

    use_class = True
    use_reg = True
    if use_class:
        importance_df_sorted_c, important_features_c, performance_df_c  = gen_results(models_c, X_train_c, y_train_c, X_test_c, y_test_c, X.columns, threshold)    

    if use_reg:
        importance_df_sorted_r, important_features_r, performance_df_r  = gen_results(models_r, X_train_r, y_train_r, X_test_r, y_test_r, X.columns, threshold)            


    print(X.shape)

    if use_class:
        print("Classifier Sorted Importance")
        print(" ")
        print(importance_df_sorted_c)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Classifier Selected Features")
        print(important_features_c)
        print("Perf Results")
        print(performance_df_c)
        print(" ")

        
    if use_reg:
        print("------------------------------------------------------------")
        print(" ")
        print("Regressor Sorted Importance")
        print(" ")
        print(importance_df_sorted_r)
        print(" ")
        print(f"Features Selected based on threshold of {threshold} percent")
        print("Regressor Selected Features")
        print(important_features_r)
        print("Perf Results")
        print(performance_df_r)
        print(" ")



if __name__ == "__main__":
    run()

"""
Model M 1
    
     Feature  LGBCls_PI_Mean  LGBCls_PI_Std  LGBCls_Std_Imp  XGBCls_PI_Mean  XGBCls_PI_Std  XGBCls_Std_Imp  CatCls_PI_Mean  CatCls_PI_Std  CatCls_Std_Imp  Comp_PI_Mean  Comp_Std_Imp
12      RSI9        0.122273       0.002964             413        0.117199       0.002247        0.710315        0.013668       0.001342       40.984910      0.084380    151.565075
24       TV5        0.001383       0.000782             201        0.010928       0.001281        0.020716        0.035735       0.002100        5.061745      0.016016     68.694154
7       ATR2        0.000056       0.000776             163        0.001702       0.001014        0.009860        0.002420       0.000850        0.965299      0.001393     54.658386
20       TV1        0.000175       0.000511             118        0.001131       0.000945        0.010354        0.002893       0.000948        1.388673      0.001399     39.799676
28     COMP2        0.001497       0.001230             111       -0.002598       0.000928        0.014329        0.001337       0.000890        5.526957      0.000078     38.847095
6       ATR5       -0.000616       0.000837             114        0.000331       0.001238        0.009677        0.001933       0.000996        0.874142      0.000549     38.294606
15     RSIL9        0.000122       0.000837             110        0.001546       0.001290        0.010334        0.006248       0.001613        2.343746      0.002639     37.451360
25       TV6       -0.000757       0.000617             106        0.000020       0.001178        0.010930        0.001284       0.001259        2.632565      0.000182     36.214498
14     RSIH9       -0.000673       0.000602              98        0.001139       0.001301        0.009081        0.002835       0.001291        1.451003      0.001100     33.153361
11    RSIL14       -0.000260       0.000791              96        0.004177       0.001187        0.010288        0.001632       0.001083        1.412444      0.001849     32.474244
26     COMP0       -0.000268       0.000794              91        0.003057       0.001273        0.014341        0.006031       0.000985        5.096081      0.002940     32.036807
23       TV4       -0.000695       0.000622              92        0.001615       0.001097        0.009217        0.002295       0.000765        1.008615      0.001072     31.005944
22       TV3       -0.000863       0.000985              91        0.001641       0.000951        0.010248        0.001314       0.001270        1.059264      0.000697     30.689837
5       ROC9       -0.001054       0.000464              89       -0.000323       0.000765        0.008814        0.002588       0.000899        0.954660      0.000404     29.987825
2     SDKC7U       -0.000614       0.000658              86        0.001949       0.001037        0.009220        0.002766       0.001003        1.060457      0.001367     29.023226
29     COMP3       -0.000362       0.000631              85        0.001243       0.001085        0.009810        0.005412       0.001310        1.315051      0.002097     28.774954
3     SDKC7L        0.001253       0.000509              85        0.002792       0.001093        0.008839        0.007413       0.001319        0.914762      0.003819     28.641200
21       TV2        0.002386       0.000867              81        0.001171       0.000916        0.010737        0.002117       0.000708        0.943832      0.001891     27.318190
1     SDBB9U        0.000433       0.000730              81        0.000940       0.001089        0.009330        0.003645       0.001442        0.913409      0.001673     27.307579
0     SDBB9L        0.000477       0.000695              78        0.005477       0.001047        0.009688        0.005617       0.000991        0.833349      0.003857     26.281012
18  STOK721D       -0.000204       0.000443              76        0.004359       0.001153        0.009428        0.008605       0.001292        0.820854      0.004254     25.610094
10    RSIH14        0.000026       0.000565              75       -0.000607       0.000956        0.009320        0.005668       0.001271        1.205489      0.001696     25.404936
8      RSI14       -0.000250       0.000674              72        0.000517       0.001149        0.011505        0.005380       0.001302        3.875590      0.001882     25.295698
19  STOK513D        0.000887       0.000593              74        0.004893       0.000946        0.008843        0.013100       0.001196        0.650644      0.006293     24.886496
4      ROC14        0.000123       0.000456              73        0.001376       0.001122        0.008488        0.001460       0.001161        1.143742      0.000987     24.717410
13   RSI9Avg       -0.000216       0.000583              64        0.008449       0.001215        0.009506        0.006571       0.001226        7.321057      0.004935     23.776854
16   STOK721       -0.001019       0.000453              70        0.001087       0.001121        0.008415        0.005510       0.001403        1.229596      0.001859     23.746004
17   STOK513       -0.000805       0.000402              61       -0.000346       0.001121        0.009386        0.003335       0.001196        0.785108      0.000728     20.598165
9   RSI14Avg        0.000481       0.000525              46        0.001126       0.001107        0.008981        0.010844       0.001289        0.700006      0.004150     15.569662
27     COMP1        0.000000       0.000000               0        0.000000       0.000000        0.000000        0.003492       0.000795        5.526951      0.001164      1.842317

Features Selected based on threshold of 50 percent
Classifier Selected Features
['RSI9', 'TV5', 'RSIL9', 'RSIL14', 'COMP0']
Perf Results
    Model      Perf  Sel_Perf        R2    Sel_R2       MSE   Sel_MSE
0  LGBCls  0.786042  0.786734  0.141056  0.143832  0.213958  0.213266
1  XGBCls  0.779918  0.782832  0.116469  0.128168  0.220082  0.217168
2  CatCls  0.786240  0.782881  0.141849  0.128366  0.213760  0.217119

------------------------------------------------------------

Regressor Sorted Importance

     Feature  LGBReg_PI_Mean  LGBReg_PI_Std  LGBReg_Std_Imp  XGBReg_PI_Mean  XGBReg_PI_Std  XGBReg_Std_Imp  CatReg_PI_Mean  CatReg_PI_Std  CatReg_Std_Imp  Comp_PI_Mean  Comp_Std_Imp
24       TV5        0.910482       0.016223             268        3.352617       0.042041        0.097579        1.276041       0.022509       11.802700      1.846380     93.300093
12      RSI9        0.398538       0.009973             221        1.564385       0.057422        0.378043        0.762436       0.011392       18.097667      0.908453     79.825237
21       TV2        0.037204       0.002598             192        0.550000       0.012441        0.015635        0.312682       0.015515        3.796331      0.299962     65.270655
23       TV4        0.044813       0.003628             178        0.371823       0.037958        0.024626        0.046577       0.004259        5.227559      0.154404     61.084062
20       TV1        0.047802       0.002726             169        0.078182       0.007624        0.023178        0.035281       0.002657        4.501281      0.053755     57.841486
8      RSI14        0.296845       0.006638             159        0.906280       0.015740        0.042493        0.604687       0.007073        7.946873      0.602604     55.663122
6       ATR5        0.003777       0.001706             142        0.014104       0.004642        0.017322        0.003422       0.002061        2.153531      0.007101     48.056951
25       TV6        0.071941       0.002054             123        0.657148       0.026553        0.024194        0.604695       0.014080        7.178272      0.444594     43.400822
7       ATR2        0.002681       0.000835             126       -0.007343       0.002698        0.019760       -0.005762       0.001366        1.654439     -0.003475     42.558066
29     COMP3        0.020335       0.001109             114        0.082411       0.003794        0.015063        0.160659       0.005503        1.558743      0.087801     38.524602
28     COMP2        0.199100       0.005422              97        3.625491       0.060467        0.040528        0.260531       0.006960        5.389469      1.361707     34.143332
22       TV3        0.015946       0.001321              97        0.083352       0.005508        0.013719        0.102148       0.006352        4.279347      0.067149     33.764355
16   STOK721        0.005564       0.001385              98        0.108452       0.003029        0.015757        0.016805       0.001468        0.642464      0.043607     32.886074
1     SDBB9U        0.001720       0.000797              88        0.160961       0.026623        0.017214        0.113037       0.006639        1.403083      0.091906     29.806766
26     COMP0        0.022486       0.001771              85        0.164447       0.006456        0.014289        0.066683       0.003678        3.488053      0.084539     29.500781
4      ROC14        0.005445       0.001437              84        0.020765       0.002262        0.009967        0.005834       0.001840        1.195422      0.010681     28.401796
3     SDKC7L        0.007607       0.001635              79        0.460873       0.007762        0.015965        1.076912       0.018222        1.638790      0.515131     26.884918
5       ROC9        0.001228       0.000531              76        0.027748       0.002909        0.011240        0.006847       0.001784        1.431922      0.011941     25.814387
18  STOK721D        0.008171       0.001049              72        0.172662       0.006410        0.017952        0.040283       0.002346        1.083495      0.073705     24.367149
14     RSIH9        0.011759       0.001112              68        0.167792       0.007264        0.020376        0.048500       0.003031        1.743155      0.076017     23.254510
11    RSIL14        0.005767       0.001165              64        0.511089       0.011569        0.019242        0.160193       0.003799        1.293055      0.225683     21.770766
15     RSIL9        0.014171       0.001149              58        0.228586       0.003761        0.014718        0.057623       0.001909        1.110817      0.100127     19.708512
2     SDKC7U        0.003299       0.000899              56        0.073171       0.003532        0.014571        0.057750       0.003146        1.149620      0.044740     19.054730
10    RSIH14        0.017337       0.001628              55        0.956381       0.016146        0.020242        0.406221       0.009294        1.407081      0.459980     18.809107
13   RSI9Avg        0.004322       0.000731              53        1.225852       0.018049        0.015308        0.129695       0.002954        2.877685      0.453290     18.630998
0     SDBB9L        0.001617       0.000695              53        0.048653       0.005090        0.008252        0.118158       0.002839        0.961209      0.056142     17.989820
17   STOK513        0.001502       0.000811              52        0.015923       0.005058        0.019009        0.054365       0.002493        1.023295      0.023930     17.680768
19  STOK513D        0.000541       0.000480              43        0.043295       0.003062        0.012124        0.044611       0.002723        0.888830      0.029482     14.633651
9   RSI14Avg        0.001231       0.000802              30        0.042421       0.002599        0.041634        0.052167       0.002868        0.973651      0.031940     10.338428
27     COMP1        0.000000       0.000000               0        0.000000       0.000000        0.000000        0.036123       0.001982        2.102161      0.012041      0.700720

Features Selected based on threshold of 50 percent
Regressor Selected Features
['TV5', 'RSI9', 'TV2', 'TV4', 'RSI14', 'TV6', 'COMP3', 'COMP2', 'SDBB9U', 'COMP0']
Perf Results
    Model    Perf  Sel_Perf        R2    Sel_R2     MSE   Sel_MSE
0  LGBReg  0.7405    0.7402  0.399334  0.396517  5.3992  5.424571
1  XGBReg  0.7394    0.7400  0.314429  0.338557  6.1624  5.945563
2  CatReg  0.7386    0.7403  0.373478  0.382910  5.6317  5.546880
    
    
    
    
    """