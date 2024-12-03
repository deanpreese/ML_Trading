import psycopg2
from psycopg2 import Error

import mlflow
import json
import pandas as pd

mlflow.set_tracking_uri(uri="http://10.0.0.50:8888")
    

# Database connection parameters
DB_HOST = '10.0.0.50'
DB_NAME = 'mlflow2'
DB_USER = 'dean'
DB_PASSWORD = 'abc'

def execute_sql_query():
    try:
        # Establish a connection to the database
        with psycopg2.connect(
            dbname='mlflow2',
            user='dean',
            password='abc',
            host='10.0.0.50'
        ) as connection:
            # Create a cursor object
            with connection.cursor() as cursor:
                # SQL query with LIMIT clause
                sql_query = """
                    SELECT r.run_uuid, r.artifact_uri, met.value
                    FROM public.runs AS r 
                    JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id 
                    JOIN public.metrics AS met ON r.run_uuid = met.run_uuid
                    WHERE exp.name NOT LIKE '%output%' 
                    AND exp.lifecycle_stage = 'active' 
                    AND met.key = 'Perf'
                    AND exp.experiment_id = 268
                    AND met.value > 0.7
                    ORDER BY met.value DESC
                    LIMIT 100
                """
                
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                return rows                    
                    
                    
    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    rows = execute_sql_query()
    
    features_list = []
    for row in rows:
        js_name = f"{row[1]}/features_used.json"
        print(js_name)
        arti_d = mlflow.artifacts.load_dict(js_name)
        features_list.append(arti_d['data'])
    
    feat_df = pd.DataFrame(features_list)    
    
    print(feat_df)
    feat_df.to_csv("features.csv")

