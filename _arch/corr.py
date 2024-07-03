
import pandas as pd
import psycopg2
from psycopg2 import sql


# Database connection parameters
dbname = 'orders'
user = 'trading'
password = 'abc'
host = '10.0.0.240'

# SDLR310,SDBB91,SDKC91,SDKC9,ROC,ATR34,ATR32,ATR31,ATR3,ATR21,ATR2,RSI,STOK1,output,outputC,actual

# List of values for the indicator parameter
values = ['sdlr310', 'sdbb91', 'scdk91', 'sdkc9', 'roc', 'atr34', 'atr32', 'atr31', 'atr3', 'atr21', 'atr2', 'rsi', 'stok1']

def main():
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    cur = conn.cursor()

    # Prepare a list to hold all results
    all_results = []

    group = 138
    max_group=148
    max_corr_len = 1100

    for corr_group in range(group, max_group, 2):
        
        for corr_len in range(100, max_corr_len, 100):

            print("Group: " + str(corr_group) + " Corr Len: " + str(corr_len))
            
            # Execute the query for each value
            for val in values:
                cur.execute(sql.SQL("SELECT * FROM get_pnl_correlation_stats(%s, %s, %s)"), [corr_group, corr_len, val])
                results = cur.fetchall()
                
                # Parse results into a list of dictionaries
                column_names = [desc[0] for desc in cur.description]
                result_list = [dict(zip(column_names, row)) for row in results]
                
                result_list[0]['group'] = corr_group
                result_list[0]['corr_len']  = corr_len

                s1 = result_list[0]['plus_plus'] + result_list[0]['min_min'] 
                s2 = result_list[0]['min_plus'] + result_list[0]['plus_min'] 
                
                result_list[0]['all_acc'] = s1 
                result_list[0]['all_miss'] = s2
                result_list[0]['per_acc'] = s1/(s1+s2) 
                all_results.extend(result_list)
            
    # Close the cursor and connection
    cur.close()
    conn.close()

    df = pd.DataFrame(all_results)
    print(df)
    
    #import json
    #with open('results.json', 'w') as f:
    #    json.dump(all_results, f, indent=4)
    
    df.to_csv('ind_corr.csv', index=False) 


if __name__ == "__main__":
    main()