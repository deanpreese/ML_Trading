
# ----------------

lucky13_all = [
                'SDLR310','SDBB91',
                'SDKC91','SDKC9',
                'ROC',
                'ATR54','ATR53',
                'ATR52','ATR51','ATR5',
                'ATR21',
                'ATR2','RSI','STOK1'
               ]

lucky_13_3070_RSI_LT_50 = ['ATR2', 'ROC', 'ATR21', 'SDKC9', 'SDBB91', 'RSI']
lucky_13_3070_RSI_GT_50 = ['RSI', 'ATR2', 'ATR21', 'ROC', 'ATR5']
lucky13_3070_comp = ['SDKC9', 'ATR5', 'ROC', 'ATR2', 'SDBB91', 'ATR21', 'RSI']
lucky13_3070_min = ['RSI', 'ATR2', 'ROC']
    
    
model_m_1_all = [
#"Year", "Month", "Day", "DayOfWeek", "HourOfDay", "MinOfHour",
#"SeqClose", "RSIRAW", 
"SDBB9L", "SDBB9U", "SDKC7U", "SDKC7L",
"ROC14", "ROC9", "ATR5", "ATR2", "RSI14", "RSI14Avg", "RSIH14", "RSIL14",
"RSI9", "RSI9Avg", "RSIH9", "RSIL9", "STOK721", "STOK513", "STOK721D", "STOK513D",
"TV1", "TV2", "TV3", "TV4", "TV5", "TV6", "COMP0", "COMP1", "COMP2", "COMP3"
]
    
    
model_m_1_alt = [
#"Year", "Month", "Day", 
"DayOfWeek", "HourOfDay", "MinOfHour",
#"SeqClose", "RSIRAW", 
#"SDBB9L", "SDBB9U", 
# "SDKC7U", "SDKC7L",
#"ROC14", 
"ROC9", 
"ATR5", 
"ATR2", 
"RSI14", 
#"RSI14Avg", 
"RSIH14", "RSIL14",
"RSI9", 
#"RSI9Avg", 
"RSIH9", "RSIL9", 
#"STOK721", 
"STOK513", 
# "STOK721D", 
#"STOK513D",
"TV1", "TV2", "TV3",
"TV4", "TV5", "TV6",
#"COMP0", 
#"COMP1", 
"COMP2", 
"COMP3"
]

model_m_1_slim = [
#"Year", "Month", "Day", 
#"DayOfWeek", "HourOfDay", "MinOfHour",
#"SeqClose", "RSIRAW", 
#"SDBB9L", "SDBB9U", 
# "SDKC7U", "SDKC7L",
#"ROC14", 
#"ROC9", 
#"ATR5", 
#"ATR2", 
"RSI14", 
#"RSI14Avg", 
"RSIH14", "RSIL14",
"RSI9", 
#"RSI9Avg", 
"RSIH9", "RSIL9", 
#"STOK721", 
"STOK513", 
# "STOK721D", 
#"STOK513D",
"TV1", "TV2", "TV3",
"TV4", "TV5", "TV6",
#"COMP0", 
#"COMP1", 
"COMP2", 
"COMP3"
]

model_m_1_slim_x = [
#"Year", "Month", "Day", 
#"DayOfWeek", "HourOfDay", "MinOfHour",
#"SeqClose", "RSIRAW", 
"SDBB9L", "SDBB9U", 
 "SDKC7U", "SDKC7L",
#"ROC14", 
"ROC9", 
"ATR5", 
"ATR2", 
"RSI14", 
#"RSI14Avg", 
"RSIH14", "RSIL14",
"RSI9", 
#"RSI9Avg", 
"RSIH9", "RSIL9", 
#"STOK721", 
"STOK513", 
# "STOK721D", 
"STOK513D",
"TV1", "TV2", 
"TV3", "TV4", 
"TV5", "TV6",
"COMP0", 
"COMP1", 
"COMP2", 
"COMP3"
]


model_m_1_r2 = [
    #"Year", "Month", "Day", 
    #"DayOfWeek", "HourOfDay", "MinOfHour",
    #"SeqClose", "RSIRAW",     
    "RSI9",
    "COMP0",
    "COMP1",
    "COMP2",
    "RSI14",
    "RSI9Avg",
    "RSIH9",
    "RSIL9",
    "TV5",
    "TV6",
    "RSIH14",
    "RSIL14",
    "TV2",
    #"COMP3",
    #"TV3",
    #"STOK513",
    #"TV1",
    #"RSI14Avg",
    #"STOK721",
    #"SDKC7L",
    #"SDKC7U",
    #"SDBB9U",
    #"SDBB9L",
    #"STOK513D",
    #"ROC14",
    #"STOK721D",
    #"ROC9",
    #"TV4",
    #"ATR5",
    #"ATR2"
]