# ============================================================
# CODTECH BIG DATA INTERNSHIP - TASK 1
# Data Cleaning with PySpark
# Name: [Your Name] | Internship: Big Data | Company: CODTECH
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, when, isnan, isnull, mean, median,
    trim, lower, regexp_replace, to_date, lit
)
from pyspark.sql.types import IntegerType, FloatType, StringType
import pyspark.sql.functions as F

# ─────────────────────────────────────────────
# Step 1: Initialize Spark Session
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("CODTECH_Task1_DataCleaning") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark Session Started Successfully!")

# ─────────────────────────────────────────────
# Step 2: Create Sample Large Dataset
# (Simulating a real-world dirty dataset)
# ─────────────────────────────────────────────
data = [
    (1,  "Alice",   28,   "F",   55000.0,  "Engineering", "2021-03-15"),
    (2,  "Bob",     35,   "M",   72000.0,  "Marketing",   "2019-07-22"),
    (3,  "Charlie", None, "M",   48000.0,  "HR",          "2020-11-01"),
    (4,  "Diana",   29,   "F",   None,     "Engineering", "2022-01-10"),
    (5,  "Eve",     42,   "F",   91000.0,  "Finance",     "2018-05-30"),
    (2,  "Bob",     35,   "M",   72000.0,  "Marketing",   "2019-07-22"),  # Duplicate
    (6,  " Frank",  31,   "M",   63000.0,  "engineering", "2021-08-19"),  # Leading space, wrong case
    (7,  "Grace",   None, "F",   54000.0,  "HR",          "2023-02-14"),
    (8,  "Hank",    27,   "M",   None,     "Finance",     "2022-09-05"),
    (9,  "Ivy",     38,   "F",   82000.0,  None,          "2020-06-17"),
    (10, "Jack",    None, "M",   None,     "Engineering", "2019-12-31"),
    (3,  "Charlie", None, "M",   48000.0,  "HR",          "2020-11-01"),  # Duplicate
    (11, "Karen",   45,   "F",   105000.0, "Finance",     "2017-03-08"),
    (12, "Leo",     33,   "M",   -5000.0,  "Marketing",   "2021-07-25"),  # Invalid salary
    (13, "Mia",     26,   "F",   58000.0,  "HR",          "2023-05-11"),
    (14, "Nate",    200,  "M",   67000.0,  "Engineering", "2022-03-22"),  # Invalid age
    (15, "Olivia",  30,   "F",   74000.0,  "Finance",     "2020-10-19"),
    (None, None,   None,  None,  None,     None,          None),          # All nulls row
]

columns = ["emp_id", "name", "age", "gender", "salary", "department", "join_date"]

df_raw = spark.createDataFrame(data, schema=columns)

print("\n📊 RAW DATASET (Before Cleaning):")
print(f"   Total Records: {df_raw.count()}")
df_raw.show(truncate=False)

# ─────────────────────────────────────────────
# Step 3: Null/Missing Value Analysis
# ─────────────────────────────────────────────
print("\n🔍 MISSING VALUE ANALYSIS:")
null_counts = df_raw.select([
    count(when(isnull(c), c)).alias(c) for c in df_raw.columns
])
null_counts.show()

# ─────────────────────────────────────────────
# Step 4: Remove Duplicate Records
# ─────────────────────────────────────────────
df_no_dupes = df_raw.dropDuplicates()
duplicates_removed = df_raw.count() - df_no_dupes.count()
print(f"\n🗑️  DUPLICATES REMOVED: {duplicates_removed} records")

# ─────────────────────────────────────────────
# Step 5: Drop rows where ALL values are null
# ─────────────────────────────────────────────
df_no_all_null = df_no_dupes.dropna(how="all")
print(f"🗑️  ALL-NULL ROWS REMOVED: {df_no_dupes.count() - df_no_all_null.count()} records")

# ─────────────────────────────────────────────
# Step 6: Fill Missing Values (Imputation)
# ─────────────────────────────────────────────
# Calculate mean salary and median age for imputation
mean_salary = df_no_all_null.select(mean("salary")).first()[0]
mean_age    = df_no_all_null.select(mean("age")).first()[0]

df_filled = df_no_all_null \
    .fillna({"salary": round(mean_salary, 2),   # Fill salary with mean
             "age":    round(mean_age),           # Fill age with mean
             "department": "Unknown",             # Fill dept with default
             "gender": "Unknown"})

print(f"\n✅ MISSING VALUES FILLED:")
print(f"   salary  → mean value: {round(mean_salary, 2)}")
print(f"   age     → mean value: {round(mean_age)}")
print(f"   department/gender → 'Unknown'")

# ─────────────────────────────────────────────
# Step 7: String Cleaning
# ─────────────────────────────────────────────
df_cleaned_str = df_filled \
    .withColumn("name",       trim(col("name"))) \
    .withColumn("department", trim(lower(col("department")))) \
    .withColumn("department", regexp_replace(col("department"), r"[^a-z\s]", ""))

# Capitalize department properly
df_cleaned_str = df_cleaned_str.withColumn(
    "department",
    F.initcap(col("department"))
)

print("\n✅ STRING CLEANING DONE (trimmed whitespace, fixed casing)")

# ─────────────────────────────────────────────
# Step 8: Remove Invalid Values (Business Rules)
# ─────────────────────────────────────────────
df_valid = df_cleaned_str \
    .filter(col("salary") >= 0) \
    .filter((col("age") >= 18) & (col("age") <= 80)) \
    .filter(col("emp_id").isNotNull())

invalid_removed = df_cleaned_str.count() - df_valid.count()
print(f"\n🚫 INVALID RECORDS REMOVED: {invalid_removed} (negative salary / invalid age / null ID)")

# ─────────────────────────────────────────────
# Step 9: Type Casting & Date Formatting
# ─────────────────────────────────────────────
df_final = df_valid \
    .withColumn("emp_id",    col("emp_id").cast(IntegerType())) \
    .withColumn("age",       col("age").cast(IntegerType())) \
    .withColumn("salary",    col("salary").cast(FloatType())) \
    .withColumn("join_date", to_date(col("join_date"), "yyyy-MM-dd"))

print("\n✅ DATA TYPES CAST CORRECTLY")

# ─────────────────────────────────────────────
# Step 10: Final Cleaned Dataset Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("📋 FINAL CLEANED DATASET")
print("="*60)
df_final.show(truncate=False)
df_final.printSchema()

print(f"\n📊 CLEANING SUMMARY:")
print(f"   Original Records   : {df_raw.count()}")
print(f"   Duplicates Removed : {duplicates_removed}")
print(f"   Invalid Removed    : {invalid_removed + 1}")  # +1 for all-null row
print(f"   Final Clean Records: {df_final.count()}")
print(f"\n✅ Task 1 Complete — Data Cleaning with PySpark!")

# ─────────────────────────────────────────────
# Step 11: Save Cleaned Data (optional)
# ─────────────────────────────────────────────
# df_final.write.mode("overwrite").csv("output/cleaned_data", header=True)
# print("💾 Cleaned data saved to output/cleaned_data/")

spark.stop()
print("\n🛑 Spark Session Stopped.")
