from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType
)

class LoadsSchema:
    schema = StructType([
        StructField("load_id", StringType(), nullable = False),
        StructField("customer_id", StringType(), nullable = True),
        StructField("route_id", StringType(), nullable = True),
        StructField("load_date", DateType(), nullable = True),
        StructField("load_type", StringType(), nullable = True),
        StructField("weight_lbs", IntegerType(), nullable = True),
        StructField("pieces", IntegerType(), nullable = True),
        StructField("revenue", DoubleType(), nullable = True),
        StructField("fuel_surcharge", DoubleType(), nullable = True),
        StructField("accessorial_charges", DoubleType(), nullable = True),
        StructField("load_status", StringType(), nullable = True),
        StructField("booking_type", StringType(), nullable = True)
    ])
