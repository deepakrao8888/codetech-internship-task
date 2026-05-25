# ============================================================
# CODTECH BIG DATA INTERNSHIP - TASK 2
# Distributed Data Processing with Apache Spark
# Name: [Your Name] | Internship: Big Data | Company: CODTECH
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, avg, max, min, round as spark_round,
    when, desc, asc, year, month, lit, countDistinct,
    percentile_approx, concat_ws, collect_list
)
from pyspark.sql.types import *
import pyspark.sql.functions as F

# ─────────────────────────────────────────────
# Step 1: Initialize Spark Session
# ─────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("CODTECH_Task2_DistributedProcessing") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark Session Initialized!")

# ─────────────────────────────────────────────
# Step 2: Load Large Sample Dataset
# (E-Commerce Sales Dataset - Simulated)
# ─────────────────────────────────────────────
sales_data = [
    (1001, "Electronics", "Laptop",      75000, 2, "North", "2024-01-15", "Online"),
    (1002, "Clothing",    "T-Shirt",       799, 5, "South", "2024-01-20", "Store"),
    (1003, "Electronics", "Smartphone",  45000, 1, "East",  "2024-02-10", "Online"),
    (1004, "Food",        "Biscuits",      120, 10,"West",  "2024-02-14", "Store"),
    (1005, "Electronics", "Laptop",      75000, 3, "North", "2024-03-05", "Online"),
    (1006, "Clothing",    "Jeans",        2499, 2, "East",  "2024-03-18", "Online"),
    (1007, "Food",        "Chips",          80, 20,"South", "2024-04-01", "Store"),
    (1008, "Electronics", "Headphones",   3500, 4, "North", "2024-04-22", "Online"),
    (1009, "Clothing",    "Jacket",       5999, 1, "West",  "2024-05-11", "Store"),
    (1010, "Food",        "Juice",         250, 8, "East",  "2024-05-30", "Online"),
    (1011, "Electronics", "Tablet",      28000, 2, "South", "2024-06-15", "Store"),
    (1012, "Clothing",    "Shirt",        1299, 3, "North", "2024-06-28", "Online"),
    (1013, "Food",        "Noodles",       150, 15,"West",  "2024-07-04", "Store"),
    (1014, "Electronics", "Smartwatch",  12000, 2, "East",  "2024-07-19", "Online"),
    (1015, "Clothing",    "Dress",        3299, 1, "South", "2024-08-08", "Store"),
    (1016, "Food",        "Biscuits",      120, 25,"North", "2024-08-20", "Online"),
    (1017, "Electronics", "Laptop",      68000, 1, "West",  "2024-09-09", "Online"),
    (1018, "Clothing",    "T-Shirt",       699, 7, "East",  "2024-09-25", "Store"),
    (1019, "Food",        "Chips",          80, 30,"South", "2024-10-13", "Store"),
    (1020, "Electronics", "Smartphone",  42000, 2, "North", "2024-10-28", "Online"),
    (1021, "Clothing",    "Jeans",        1999, 4, "West",  "2024-11-05", "Online"),
    (1022, "Food",        "Juice",         300, 12,"East",  "2024-11-18", "Store"),
    (1023, "Electronics", "Headphones",   4200, 3, "South", "2024-12-02", "Online"),
    (1024, "Clothing",    "Jacket",       7999, 2, "North", "2024-12-20", "Store"),
    (1025, "Food",        "Noodles",       200, 20,"West",  "2024-12-29", "Online"),
]

schema = StructType([
    StructField("order_id",   IntegerType(), True),
    StructField("category",   StringType(),  True),
    StructField("product",    StringType(),  True),
    StructField("price",      IntegerType(), True),
    StructField("quantity",   IntegerType(), True),
    StructField("region",     StringType(),  True),
    StructField("order_date", StringType(),  True),
    StructField("channel",    StringType(),  True),
])

df = spark.createDataFrame(sales_data, schema=schema)

# Add revenue column
df = df.withColumn("revenue", col("price") * col("quantity")) \
       .withColumn("order_date", F.to_date(col("order_date"), "yyyy-MM-dd")) \
       .withColumn("month", month(col("order_date")))

print("\n📦 RAW SALES DATASET:")
print(f"   Total Orders: {df.count()} | Partitions: {df.rdd.getNumPartitions()}")
df.show(5)

# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: FILTERING — High-Value Orders (Revenue > 50,000)
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 1: HIGH-VALUE ORDERS (Revenue > ₹50,000)")
print("─"*60)
df_high_value = df.filter(col("revenue") > 50000) \
                  .select("order_id", "category", "product", "price", "quantity", "revenue") \
                  .orderBy(desc("revenue"))
df_high_value.show()
print(f"   → Total High-Value Orders: {df_high_value.count()}")

# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: FILTERING — Online Channel Orders Only
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 2: ONLINE VS STORE ORDERS")
print("─"*60)
df_channel = df.groupBy("channel") \
               .agg(
                   count("order_id").alias("total_orders"),
                   spark_round(sum("revenue"), 2).alias("total_revenue"),
                   spark_round(avg("revenue"), 2).alias("avg_revenue")
               ).orderBy(desc("total_revenue"))
df_channel.show()

# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: GROUPING — Revenue by Category
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 3: REVENUE GROUPED BY CATEGORY")
print("─"*60)
df_by_category = df.groupBy("category") \
                   .agg(
                       count("order_id").alias("total_orders"),
                       sum("quantity").alias("total_units_sold"),
                       spark_round(sum("revenue"), 0).alias("total_revenue"),
                       spark_round(avg("revenue"), 2).alias("avg_order_value"),
                       max("revenue").alias("max_single_order")
                   ).orderBy(desc("total_revenue"))
df_by_category.show()

# ══════════════════════════════════════════════════════════════
# ANALYSIS 4: GROUPING — Performance by Region
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 4: REGIONAL PERFORMANCE")
print("─"*60)
df_by_region = df.groupBy("region") \
                 .agg(
                     count("order_id").alias("orders"),
                     spark_round(sum("revenue"), 0).alias("total_revenue"),
                     spark_round(avg("revenue"), 2).alias("avg_revenue")
                 ).orderBy(desc("total_revenue"))
df_by_region.show()

# ══════════════════════════════════════════════════════════════
# ANALYSIS 5: AGGREGATION — Monthly Sales Trend
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 5: MONTHLY REVENUE AGGREGATION")
print("─"*60)
df_monthly = df.groupBy("month") \
               .agg(
                   count("order_id").alias("orders"),
                   spark_round(sum("revenue"), 0).alias("monthly_revenue")
               ).orderBy("month")
df_monthly.show(12)

# ══════════════════════════════════════════════════════════════
# ANALYSIS 6: AGGREGATION — Top 5 Best-Selling Products
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 6: TOP 5 BEST-SELLING PRODUCTS")
print("─"*60)
df_top_products = df.groupBy("product", "category") \
                    .agg(
                        sum("quantity").alias("total_units"),
                        spark_round(sum("revenue"), 0).alias("total_revenue")
                    ).orderBy(desc("total_revenue")) \
                    .limit(5)
df_top_products.show()

# ══════════════════════════════════════════════════════════════
# ANALYSIS 7: SPARK SQL — Using SQL Queries on DataFrame
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("📌 ANALYSIS 7: SPARK SQL — Category vs Region Cross-Analysis")
print("─"*60)
df.createOrReplaceTempView("sales")

sql_result = spark.sql("""
    SELECT 
        category,
        region,
        COUNT(order_id)    AS orders,
        SUM(revenue)       AS total_revenue,
        ROUND(AVG(revenue), 2) AS avg_revenue
    FROM sales
    GROUP BY category, region
    ORDER BY category, total_revenue DESC
""")
sql_result.show(20)

# ══════════════════════════════════════════════════════════════
# ANALYSIS 8: WINDOW FUNCTION — Rank products within category
# ══════════════════════════════════════════════════════════════
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

print("\n" + "─"*60)
print("📌 ANALYSIS 8: WINDOW FUNCTION — Product Rank within Category")
print("─"*60)
window_spec = Window.partitionBy("category").orderBy(desc("revenue"))
df_ranked = df.withColumn("rank", rank().over(window_spec)) \
              .filter(col("rank") == 1) \
              .select("category", "product", "revenue", "rank")
df_ranked.show()

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("📊 ANALYSIS COMPLETE — SUMMARY")
print("="*60)
print(f"  Total Orders Analyzed    : {df.count()}")
print(f"  Total Revenue (all)      : ₹{df.agg(sum('revenue')).first()[0]:,.0f}")
print(f"  Avg Order Value          : ₹{df.agg(avg('revenue')).first()[0]:,.2f}")
print(f"  Max Single Order Revenue : ₹{df.agg(max('revenue')).first()[0]:,.0f}")
print(f"\n✅ Task 2 Complete — Distributed Data Processing with Spark!")

spark.stop()
print("🛑 Spark Session Stopped.")
