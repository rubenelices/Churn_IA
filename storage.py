import pandas as pd

url = "https://storage.googleapis.com/jvelare-public/bq-results-20260211-093843-1770799837095"
customer_data = pd.read_csv(url)
customer_data.to_csv("customer_data.csv", index=False)