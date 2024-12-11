from collections import Counter



x_LT_30_50 = ['TV6', 'TV1', 'STO7143D', 'ATR14', 'SDBB9CL', 'SDKC7CU', 'SDKC10CL', 'ATR5', 'ZH79X', 'COMP3', 'SDKC9', 'TV3', 
                    'SDBB9', 'ATR2', 'STO7143K', 'ZC79X', 'TV2', 'ROC9', 'COMP2', 'ROC14', 'STO5135K', 'STO5135D', 'RSIRAW']

x_LT_30_75 = ['RSIRAW', 'ROC14', 'TV1', 'COMP2', 'SDKC9', 'ROC9']



x_GT_70_75 = ['SDKC9', 'ZH79X', 'ROC14', 'COMP3', 'RSIRAW', 'COMP2', 'ATR2', 
                            'STO7143D', 'TV6', 'TV3', 'SDKC7CU', 'ROC9', 'ZC79X', 'ATR14', 'TV1', 'STO5135K']

x_GT_70_50 = ['ZC79X', 'SDKC9', 'MinOfHour', 'SDBB14CU', 'ADX14', 'ATR14', 'SDKC10CU', 'ATR2', 
                            'SDKC7CU', 'TV6', 'COMP2', 'STO5135K', 'RSIRAW', 'STO5135D', 'COMP3', 'TV4', 'ADX9', 'TV1', 
                            'STO7143D', 'ZH79X', 'HourOfDay', 'ROC7', 'SeqClose']


x_GT_50_75 = ['SDKC9', 'COMP2', 'ADX14', 'RSIRAW', 'SDKC91', 'SeqClose', 'TV3', 'STO5135D', 'TV4', 'SDKC7CU']

x_GT_50_50 = ['TV5', 'ROC9', 'ATR14', 'TV4', 'COMP2', 'SDKC7CL', 'SDKC9', 'TV6', 'HourOfDay', 'ZL79X', 
                            'MinOfHour', 'ZH79X', 'RSIRAW', 'ROC14', 'ATR2', 'TV1', 'ZC79X', 'TV3', 'SDKC91', 'STO5135D', 'SDBB9CU', 
                            'SDKC7CU', 'ADX14', 'SeqClose']

x_LT_50_75 = ['RSIRAW', 'ZH79X', 'SDKC9', 'SDKC7CU', 'COMP2']

x_LT_50_50 =  ['TV4', 'SDBB9CL', 'TV1', 'COMP3', 'TV3', 'ZH79X', 'STO7143D', 'HourOfDay', 'ATR2', 'SDKC7CL', 'ADX14', 'SDKC9', 
                        'SeqClose', 'ATR5', 'Day', 'ADX9', 'SDKC10CL', 'ROC9', 'MinOfHour', 'STO5135D', 'RSIRAW', 'SDKC7CU', 
                        'ZC79X', 'ROC14', 'COMP2', 'TV6']

model_x_3070_imp_full =['SDKC7CU', 'ZL79X', 'TV3', 'ROC14', 'ATR2', 'STO5135D', 
                        'STO7143D', 'TV4', 'SDKC91', 'ZC79X', 'ATR9', 'TV5', 
                        'RSIRAW', 'COMP3', 'SDBB9CL', 'SDKC9', 'TV2', 'TV6', 
                        'SDKC7CL', 'COMP0', 'ZH79X', 'SDBB91', 'TV1', 'COMP2']

model_x_3070_imp_slim = ['SDKC9', 'COMP3', 'STO7143D', 'ATR9', 'SDBB91', 'RSIRAW', 'COMP2', 'TV6', 'SDKC7CU']





# Combine the lists
combined_list = model_x_3070_imp_full + model_x_3070_imp_slim + x_LT_30_50 + x_LT_30_75 + x_GT_70_75 + x_GT_70_50 +x_GT_50_75 + x_GT_50_50 + x_LT_50_50 + x_LT_50_75

# Count the frequency of each item
item_counts = Counter(combined_list)

# Sort by count (descending) and then by item (alphabetically)
sorted_item_counts = sorted(item_counts.items(), key=lambda x: (-x[1], x[0]))

# Print the sorted counts
print(f" Features used  {len(sorted_item_counts)} ")
print("Item counts:")
for item, count in sorted_item_counts:
    print(f"{item}: {count}")