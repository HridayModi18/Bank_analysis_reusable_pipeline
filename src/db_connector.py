import os
import pymysql
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


load_dotenv()

DB_HOST = os.getenv("DB_HOST")

DB_PORT = int(os.getenv("DB_PORT"))

DB_USER = os.getenv("DB_USER")

DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_NAME = os.getenv("DB_NAME")

def get_connection():
    """Establish a raw connection to the MySQL database."""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def get_sqlalchemy_engine():
    connection_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_url)

def run_data_quality_procedure():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            # Execute cleaning stored procedure
            cursor.execute("CALL sp_clean_and_validate_data()")
        connection.commit()
        print("Data qualityStored procedure executed successfully.")
    except Exception as error:
        print(f"Error executing stored procedure: {error}")
        raise error
    finally:
        connection.close()

def fetch_analytics_features():
    engine = get_sqlalchemy_engine()
    query = "SELECT * FROM vw_Analytics_Features"
    try:
        df = pd.read_sql(query, con=engine)
        print(f"Retrieved {len(df)} records from vw_Analytics_Features")
        return df
    except Exception as error:
        print(f"Error fetching analytics features: {error}")
        raise error
def fetch_data_quality_metrics():
    engine = get_sqlalchemy_engine()
    query = "SELECT * FROM vw_Data_Quality_Report"
    try:
        df = pd.read_sql(query, con=engine)
        print("Retrieved data quality metrics")
        return df
    except Exception as error:
        print(f"Error fetching data quality metrics: {error}")
        raise error
