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

# PostgreSQL query
query_r2 = """
SELECT m.run_uuid
FROM metrics AS m
JOIN public.runs AS r ON r.run_uuid = m.run_uuid
JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id
WHERE exp.experiment_id > 0
  AND exp.name NOT LIKE '%output%'
  AND exp.lifecycle_stage = 'active'
  AND m.key LIKE 'Perf'
ORDER BY m.value DESC
LIMIT 5;
"""

query_perf = """
SELECT m.run_uuid
FROM metrics AS m
JOIN public.runs AS r ON r.run_uuid = m.run_uuid
JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id
WHERE exp.experiment_id > 0
  AND exp.name NOT LIKE '%output%'
  AND exp.lifecycle_stage = 'active'
  AND m.key LIKE 'Perf'
ORDER BY m.value DESC
LIMIT 5;
"""


def fetch_r2():
    return fetch_run_uuid_data(query_r2)

def fetch_perf():
    return fetch_run_uuid_data(query_perf)


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
