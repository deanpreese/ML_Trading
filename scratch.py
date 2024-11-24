import pandas as pd

#one = ['TV1', 'ZL79X', 'COMP0', 'ZL57X', 'HourOfDay', 'ATR2', 'SDLR310', 'CATR3', 'TV3', 'ROC91', 'ROC71', 'ZC79X', 'SDBB29C', 'SDBB91', 'CRSI14', 'SDKC29C', 'ZH79X', 'SDKC9C', 'TV6']
#two = ['ZH57X', 'ZL79X', 'TV3', 'ZH79X', 'ATR2', 'ZC79X', 'COMP3', 'RSI141', 'ROC14', 'ROC141', 'ROC9', 'ATR51', 'STOK51331', 'ZC911X', 'ATR5', 'SDBB291C', 'SDBB91C', 'ROC71']
#three = ['RSI9', 'TV6', 'TV1', 'COMP0', 'RSI14', 'ATR2', 'HourOfDay', 'ZC79X', 'CATR2', 'ZH911X', 'ZH57X', 'ZC57X', 'ZL911X', 'Day', 'CRSI3', 'ZH79X', 'RSI141', 'STOK5133', 'ROC14', 'ROC141', 'SDLR310', 'SDKC9C', 'ROC7']
#four = ['RSI9', 'ATR2', 'RSI14', 'COMP2', 'ZL57X', 'ATR5', 'ZH57X', 'ATR21', 'COMP0', 'STOK51331', 'TV5', 'TV6', 'COMP3', 'SDKC29C', 'ATR14', 'SDLR310V', 'ROC141', 'SDKC9', 'TV2', 'SDKC291C', 'ROC14', 'SDBB9', 'ATR51']

one =  ['ZH57X', 'ATR2', 'ROC14', 'ROC9', 'ATR51']
two =  ['TV1', 'COMP0', 'HourOfDay', 'ROC91', 'ROC71', 'ZC79X']
three =  ['TV1', 'ZL79X', 'COMP0', 'ZL57X', 'HourOfDay', 'ATR2', 'SDLR310', 'CATR3', 'TV3', 'ROC91', 'ROC71', 'ZC79X', 'SDBB29C', 'SDBB91', 'CRSI14', 'SDKC29C', 'ZH79X', 'SDKC9C', 'TV6']
four = ['ZH57X', 'ZL79X', 'TV3', 'ZH79X', 'ATR2', 'ZC79X', 'COMP3', 'RSI141', 'ROC14', 'ROC141', 'ROC9', 'ATR51', 'STOK51331', 'ZC911X', 'ATR5', 'SDBB291C', 'SDBB91C', 'ROC71']
 

#full_list = one + two + three + four
#items = list(set(full_list))
#print(items)


#   ['SDKC9C', 'ATR21', 'SDBB29C', 'ZL79X', 'SDBB91C', 'ROC141', 'TV5', 'ROC71', 'TV1', 'COMP2', 'ZC57X', 
#  'SDKC291C', 'SDLR310', 'STOK51331', 'SDBB91', 'ROC14', 'SDKC29C', 'TV6', 'SDBB291C', 'RSI141', 'SDBB9', 
# 'ZH57X', 'CATR2', 'ZH79X', 'RSI9', 'ZC911X', 'ATR5', 'ZL57X', 'RSI14', 'ROC7', 'ATR14', 'ROC9', 'CRSI3', 
# 'ATR51', 'COMP0', 'ZC79X', 'TV3', 'CRSI14', 'COMP3', 'ZL911X', 'SDKC9', 'HourOfDay', 'SDLR310V', 'STOK5133', 
# 'Day', 'ROC91', 'ATR2', 'CATR3', 'TV2', 'ZH911X']


final_list = ['TV6', 'RSI141', 'TV1', 'CATR3', 'ATR2', 'ZH79X', 'TV3', 'ATR51', 'SDBB29C', 'SDKC9C', 'ZL79X', 
 'ZC911X', 'STOK51331', 'COMP3', 'HourOfDay', 'SDLR310', 'SDKC29C', 'COMP0', 'CRSI14', 'ZH57X', 
 'SDBB91', 'SDBB291C', 'ROC71', 'ATR5', 'ROC9', 'ROC91', 'ZC79X', 'ROC141', 'ZL57X', 'SDBB91C', 'ROC14']


kan_list = [
    'RSI9', 'RSI14', 'COMP2', 'COMP1', 'COMP0', 'TV5', 'TV6', 
    'RSI91', 'ROC9', 'COMP3', 'STOK5133', 'ROC7', 'STOK714Y', 
    'ROC14', 'RSI141', 'TV2', 'TV3', 'ROC141', 'ROC91', 
    'TV1', 'ROC71', 'STOK714Y1', 'STOK51331', 'TV4'
]

#big_list = final_list + kan_list
#big_items = list(set(big_list))

one = ['RSI9', 'ATR2', 'RSI14', 'TV6', 'TV1', 'ZL57X', 'COMP0', 'HourOfDay', 'SDBB91', 'ZH79X']
two = ['RSI9', 'ATR2', 'ATR5', 'ATR51', 'RSI14', 'TV3', 'TV6', 'COMP2', 'SDKC29C', 'COMP3']

list_x = one + two
f_items = list(set(list_x))

print(len(list_x))
print(f_items)

