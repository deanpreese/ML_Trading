import psycopg2

def get_correlation(storer_id, conn_params):
    # Connection parameters - replace these with your details
    
    # SQL command with a placeholder
    sql = """
    SELECT 
        corr("ROC"::numeric, "PNL") as corr_ROC_PNL,
        corr("RSI"::numeric, "PNL") as corr_RSI_PNL,
        corr("ATR2"::numeric, "PNL") as corr_ATR2_PNL,
        corr("ATR3"::numeric, "PNL") as corr_ATR3_PNL,
        corr("ATR21"::numeric, "PNL") as corr_ATR21_PNL,
        corr("ATR31"::numeric, "PNL") as corr_ATR31_PNL,
        corr("ATR32"::numeric, "PNL") as corr_ATR32_PNL,
        corr("ATR34"::numeric, "PNL") as corr_ATR34_PNL,
        corr("SDKC9"::numeric, "PNL") as corr_SDKC9_PNL,
        corr("STOK1"::numeric, "PNL") as corr_STOK1_PNL,
        corr("SDBB91"::numeric, "PNL") as corr_SDBB91_PNL,
        corr("SDKC91"::numeric, "PNL") as corr_SDKC91_PNL,
        corr("SDLR310"::numeric, "PNL") as corr_SDLR310_PNL
    FROM  (
        SELECT 
            "PNL",
            "StorerID",	
            "OpenFeatureData" ->> 'ROC' as "ROC",
            "OpenFeatureData" ->> 'RSI' as "RSI",
            "OpenFeatureData" ->> 'ATR2' as "ATR2",
            "OpenFeatureData" ->> 'ATR3' as "ATR3",
            "OpenFeatureData" ->> 'ATR21' as "ATR21",
            "OpenFeatureData" ->> 'ATR31' as "ATR31",
            "OpenFeatureData" ->> 'ATR32' as "ATR32",
            "OpenFeatureData" ->> 'ATR34' as "ATR34",
            "OpenFeatureData" ->> 'SDKC9' as "SDKC9",
            "OpenFeatureData" ->> 'STOK1' as "STOK1",
            "OpenFeatureData" ->> 'SDBB91' as "SDBB91",
            "OpenFeatureData" ->> 'SDKC91' as "SDKC91",
            "OpenFeatureData" ->> 'SDLR310' as "SDLR310"
        FROM public."ClosedTradeLog"
        where "StorerID" <= %s 
    ) as features;    
    """
    
    # Connect to the database
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    try:
        # Execute the query with the storer_id parameter
        cur.execute(sql, (storer_id,))
        results = cur.fetchall()
        return results
    finally:
        cur.close()
        conn.close()




conn_params = {
    'dbname': 'orders',
    'user': 'dean',
    'password': 'abc',
    'host': 'localhost',
    'port': '5432'
}


mm_conn = psycopg2.connect(**conn_params)
mm_cur = mm_conn.cursor()

mm_cur.execute("SELECT MIN(\"StorerID\")  FROM public.\"ClosedTradeLog\";")
min_id = mm_cur.fetchone()
mm_cur.execute("SELECT MAX(\"StorerID\") FROM public.\"ClosedTradeLog\";")
max_id = mm_cur.fetchone()

##print(f"{min_id} {max_id}")

for sto_id in range(min_id[0],max_id[0],1):
    corr_results = get_correlation(sto_id, conn_params)
   
    mm_cur.execute(f"SELECT \"PNL\" FROM public.\"ClosedTradeLog\" WHERE \"StorerID\" = {sto_id};")
    corr_pnl = mm_cur.fetchone()
    
    sql_ins = """
    INSERT INTO public.corrfeaturedata (
        storerid, pnl, roc, rsi, atr2, atr3, atr21, atr31, atr32, atr34, sdkc9, stok1, sbdd91, sdkc91, sdlr310
    ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    data = []
    data.append(sto_id)
    data.append(corr_pnl[0])        
    
    for i in range(len(corr_results[0])):
        data.append(corr_results[0][i])
        
    print(f"{data}")
    
    mm_cur.execute(sql_ins, data)
    mm_conn.commit()  
    
    
mm_cur.close()
mm_conn.close()    
