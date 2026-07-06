import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv(
    "/Users/hridaymodi/Downloads/CreditCard/data/bank_churn_ascii.csv"
)

engine = create_engine(
    "mysql+pymysql://root:hriday7580@localhost:3306/creditcard"
)

df.to_sql(
    "bankchurn",
    con=engine,
    if_exists="replace",
    index=False
)

print("Imported successfully!")
print("Rows:", len(df))