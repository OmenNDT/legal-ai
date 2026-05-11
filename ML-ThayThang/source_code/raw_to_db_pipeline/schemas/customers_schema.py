from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType,
)

class CustomersSchema:
    schema = StructType([
        StructField("customer_id", StringType(), nullable = False),
        StructField("customer_name", StringType(), nullable = True),
        StructField("customer_type", StringType(), nullable = True),
        StructField("credit_terms_days", IntegerType(), nullable = True),
        StructField("primary_freight_type", StringType(), nullable = True),
        StructField("account_status", StringType(), nullable = True),
        StructField("contract_start_date", DateType(), nullable = True),
        StructField("annual_revenue_potential", DoubleType(), nullable = True)
    ])
