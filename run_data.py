
import pandas as pd
import psycopg2
from psycopg2 import sql

# Database connection details (replace with your actual credentials)
dbname = 'mlflow2'
user = 'dean'
password = 'abc'
host = '10.0.0.50'

def execute_query(sql):

    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        print(rows)

        cur.close()
        conn.close()

        return rows

    except Exception as e:
        print("Error:", e)
        return None

# Replace with your actual SQL query
sql = """
SELECT m.run_uuid
FROM metrics AS m
JOIN public.runs AS r ON r.run_uuid = m.run_uuid
JOIN public.experiments AS exp ON r.experiment_id = exp.experiment_id
WHERE exp.experiment_id > 0
  AND exp.name NOT LIKE '%output%'
  AND exp.lifecycle_stage = 'active'
  AND m.key LIKE 'Perf'
ORDER BY value DESC
LIMIT 5;
"""

# Execute the query and get the results
results = execute_query(sql)

# Print the results (modify as needed to process the data)
if results:
    for row in results:
        print(row[0])  # Access the run_uuid in the first column
else:
    print("No results found.")