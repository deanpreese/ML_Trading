
def lucky_features():

    features_full = [
        'RSI',
        'STOK1',
        'SDKC9',
        'SDLR310',
        'ATR2',
        'SDKC91',
        'SDBB91',
        'ATR3',
        'ATR21',
        'ATR31',
        'ROC',
        'ATR32',
        'ATR34'
    ]


    features_87_FI = [
        'RSI',
        'STOK1',
        'SDKC9',
        'SDLR310',
        'ATR2',
        'SDKC91',
        'SDBB91',
        'ATR3',
        'ATR21',
    ]


    feat_list=  [
    ['STOK1', 'SDKC9', 'SDLR310'],			
    ['STOK1', 'SDLR310', 'ATR21'],			
    ['STOK1', 'SDLR310', 'ATR3'],			
    ['STOK1', 'SDLR310', 'ATR2'],			
    ['STOK1', 'SDLR310', 'SDKC91'],			
    ['STOK1', 'SDLR310', 'SDBB91'],			
    ['STOK1', 'SDKC91', 'SDBB91'],			
    ['STOK1', 'SDKC9', 'SDBB91'],			
    ['STOK1', 'SDBB91', 'ATR3'],		
    ['STOK1', 'SDKC9', 'SDKC91'],			
    ['STOK1', 'ATR2', 'SDBB91'],			
    ['STOK1', 'SDBB91', 'ATR21'],			
    ['STOK1', 'ATR2', 'SDKC91'],		
    ['STOK1', 'SDKC9', 'ATR3'],			
    ['STOK1', 'ATR2', 'ATR21'],			
    ['STOK1', 'ATR2', 'ATR3'],			
    ['STOK1', 'SDKC9', 'ATR2'],			
    ['STOK1', 'SDKC9', 'ATR21'],			
    ['STOK1', 'SDKC9', 'SDLR310', 'SDKC91', 'ATR3', 'ATR21'],			
    ['STOK1', 'ATR3', 'ATR21'],			
    ['STOK1', 'SDKC9', 'SDLR310', 'ATR2', 'SDKC91', 'ATR21'],			
    ['STOK1', 'SDKC91', 'ATR21'],			
    ['STOK1', 'SDKC91', 'ATR3'],			
    ['STOK1', 'SDKC9', 'SDLR310', 'ATR2', 'SDKC91', 'ATR3'],			
    ['STOK1', 'SDKC9', 'SDLR310', 'SDBB91', 'ATR3', 'ATR21']
    ]
    
    return feat_list