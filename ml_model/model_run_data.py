import psycopg2
from psycopg2 import sql

# Database connection parameters
db_params = {
    'dbname': 'mlflow2',    # replace with your database name
    'user': 'dean',          # replace with your database user
    'password': 'abc',  # replace with your database password
    'host': '10.0.0.50',          # replace with your host, e.g., 'localhost'
    'port': '5432'           # replace with your port, e.g., '5432'
}

def fetch_data(num_models, asc_desc, exp_query, features, perf_r2):
    
    query = f"""
    select  
    	m.run_uuid 
	from metrics as m
	JOIN public.params as pm on pm.run_uuid = m.run_uuid
	JOIN public.runs AS r ON r.run_uuid = m.run_uuid	
	JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id

	WHERE exp.experiment_id {exp_query}
		AND exp.name NOT LIKE '%output%' 
	    AND exp.lifecycle_stage = 'active'
		and ( m.key like {perf_r2})
		and ( pm.key like 'FeatureCount')
		and CAST(pm.value as INTEGER) > {features}
		ORDER BY m.value {asc_desc}
		LIMIT {num_models};
    """

    return fetch_run_uuid_data(query)

def fetch_run_uuid_data(query_name):
    
    run_data = []
    
    try:
        # Connect to the PostgreSQL database
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query_name)
                results = cursor.fetchall()
                for row in results:
                    run_data.append(row[0])   

    except Exception as error:
        print(f"Error: {error}")

    return run_data
