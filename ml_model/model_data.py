import psycopg2

# Database connection parameters
db_params = {
    'dbname': 'mlfow2',
    'user': 'mlflow',
    'password': 'abc',
    'host': '10.0.0.50',  # e.g., 'localhost' or IP address
    'port': '5432'   # e.g., '5432'
}

# PostgreSQL query
query = """
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

def fetch_data():
    
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(**db_params)
        
        print("Connected to the PostgreSQL database " , connection )
        
        cursor = connection.cursor()

        # Execute the query
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchall()

        # Print the results
        for row in results:
            print(row)

    except Exception as error:
        print(f"Error: {error}")
    
    finally:
        # Close the database connection
        if connection:
            cursor.close()
            connection.close()

if __name__ == '__main__':
    fetch_data()
