import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import boto3
import json
import array
import numpy as np
from collections import Counter

from botocore.client import Config

def fetch_data(min_feature, max_feature, r_limit, group_limit):
    # Database connection parameters
    conn_params = {
        "host": "10.0.0.50",
        "database": "mlflowdb",
        "user": "dean",
        "password": "abc"
    }
    
    
    # SQL query as provided
    sql_query = f"""
    WITH exp_data AS 
    (
        SELECT 
            e.run_uuid,
            e.artifact_uri,
            expi.experiment_id,
            expi.name
        FROM public.runs AS e 
        JOIN public.experiments AS expi ON e.experiment_id = expi.experiment_id
        WHERE expi.name  NOT LIKE '%output%'
    ),
    met_data AS
    (
        SELECT 
            run_uuid,     
            value
        FROM public.metrics 
        WHERE key = 'Perf'
    ),
    param_data AS    
    (
        SELECT 
            run_uuid, 
            value as features
        FROM public.params
        WHERE key = 'FeatureCount'
    )

    SELECT
        e.run_uuid,
        e.name,
        e.experiment_id,
        m.value AS perf,
        e.artifact_uri,    
        (p.features::numeric) as featurecount
    FROM
        exp_data e
    JOIN
        met_data m ON e.run_uuid = m.run_uuid
    JOIN
        param_data p ON e.run_uuid = p.run_uuid
    WHERE m.value > 0.75    
        AND e.experiment_id > {group_limit}
        AND (p.features::numeric) > {min_feature}
        AND (p.features::numeric) < {max_feature}
    ORDER BY (m.value) DESC  
    LIMIT {r_limit};
    """

    # Connect to the database
    try:
        conn = psycopg2.connect(**conn_params)
        # Fetch data into DataFrame
        
        
        
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def list_s3_contents(s3_uri, minio_url, access_key, secret_key, secure=True):

    assert s3_uri.startswith('s3://'), "S3 URI must start with 's3://'"
    s3_path = s3_uri[5:]  # remove 's3://'
    bucket_name, prefix = s3_path.split('/', 1)

    # Initialize a boto3 client for Minio
    s3 = boto3.client('s3',
                      endpoint_url=minio_url,
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      config=Config(signature_version='s3v4'),
                      region_name='us-east-1',  # Optional: Depends on Minio config
                      verify=secure)  # Set to False if your Minio is not using SSL

    # Use paginator to handle buckets with many objects
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    obj_list  = []

    # Loop through pages and list objects
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                
                if obj['Key'].endswith('.json'):
                    #print(obj['Key'])
                    
                    response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    content = response['Body'].read().decode('utf-8')
                    json_data = json.loads(content)
                    obj_list.append(json_data)

    return obj_list


def get_features(min_f, max_f, r_lim, grp_lim):
    
    data = fetch_data(min_f, max_f, r_lim, grp_lim)
    
    #data.to_csv('coredata.csv', index=False, mode='a') 
    
    distinct_list = []
    run_ids = []

    for i in range(len(data['artifact_uri'])):
        
        rid = data.iloc[i][0]
        run_ids.append(rid)
        
        full_uri = data.iloc[i][4]
        j_list = list_s3_contents(full_uri, 'http://10.0.0.50:9000', aws_access_key_id, aws_secret_access_key, secure=True)
        
        df =pd.DataFrame(j_list)
        #df.to_csv('jsondata.csv', index=False, mode='a') 
        
        features = []
        
        for item in j_list:
            for ind in item['data']:
                a1 = np.array(ind)
                features.append(a1[0])
                
        #print(f"RID {data.iloc[i][0]} {features}")
        distinct_list.append(features)
        
    df = pd.DataFrame(distinct_list).drop_duplicates()

    cnt = 0
    set_a = []

    for row in df.iterrows():
    
        if cnt == 0:
            set_a = set(np.array(row[1]))
            cnt = 1
        else:
            #print(f"R  {set_a}   ")
            if set_a.issubset(np.array(row[1])):
                df.drop(row[0], inplace=True)
            
            set_a = set(np.array(row[1]))
        
    print(df)

    id_list =  [] 

    for id in df[0].index:
        print(f" {id}  {run_ids[id]}")
        id_list.append(run_ids[id])

    print(f"Unique sets {len(df)}"  )
    print(" ")
    
    return id_list
    

aws_access_key_id='minioadmin'
aws_secret_access_key='minioadmin'

f_idx = []

f_idx = get_features(1, 3, 1,30)
f_idx += get_features(2, 4, 1,30)
f_idx += get_features(3, 5, 1,30)
f_idx += get_features(4, 6, 1,50)
f_idx += get_features(5, 7, 1,50)
f_idx += get_features(6, 8, 1,60)
f_idx += get_features(7, 9, 1,60)
f_idx += get_features(8, 10, 1,60)

print(" ")
print(f_idx)
print(" ")        
        
        
