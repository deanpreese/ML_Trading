import pandas as pd

x1 =['RSIRAW', 'SeqClose', 'ADX14', 'TV6', 'ROC14', 'TV4', 'SDKC7CU', 'STO5135D', 'TV3', 'HourOfDay', 'ZL79X', 'ATR14', 'ROC9', 'ZC79X', 'ZH79X', 'SDBB9CU', 'SDKC9']
x2 = ['RSIRAW', 'ATR2', 'SDKC9', 'TV3', 'SDKC91', 'COMP2', 'ADX14', 'MinOfHour', 'TV5', 'STO5135D', 'ROC9', 'HourOfDay', 'TV1', 'SDKC7CU', 'SDKC7CL']

combined_unique_all = list(set(x1 + x2))
print(combined_unique_all)
