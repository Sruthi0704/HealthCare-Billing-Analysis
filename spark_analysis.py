from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg

spark = SparkSession.builder.appName("Healthcare Analysis").getOrCreate()

df = spark.read.csv("data/final_healthcare_billing.csv", header=True, inferSchema=True)

# Clean column names
df = df.toDF(*[c.replace(" ", "_") for c in df.columns])

print("Total Revenue:")
df.select(sum("Billing_Amount")).show()

print("Average Stay:")
df.select(avg("Length_of_Stay")).show()

spark.stop()
