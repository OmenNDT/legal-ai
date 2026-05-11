from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, TimestampType
)

class FuelPurchasesSchema:
    schema = StructType([
        StructField("fuel_purchase_id", StringType(), nullable = False),
        StructField("trip_id", StringType(), nullable = True),
        StructField("truck_id", StringType(), nullable = True),
        StructField("driver_id", StringType(), nullable = True),
        StructField("purchase_date", TimestampType(), nullable = True),
        StructField("location_city", StringType(), nullable = True),
        StructField("location_state", StringType(), nullable = True),
        StructField("gallons", DoubleType(), nullable = True),
        StructField("price_per_gallon", DoubleType(), nullable = True),
        StructField("total_cost", DoubleType(), nullable = True),
        StructField("fuel_card_number", StringType(), nullable = True)
    ])
