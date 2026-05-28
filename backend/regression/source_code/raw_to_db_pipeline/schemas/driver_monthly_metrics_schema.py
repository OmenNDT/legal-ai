from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType
)

class DriverMonthlyMetricsSchema:
    schema = StructType([
        StructField("driver_id", StringType(), nullable = False),
        StructField("month", DateType(), nullable = False),
        StructField("trips_completed", IntegerType(), nullable = True),
        StructField("total_miles", DoubleType(), nullable = True),
        StructField("total_revenue", DoubleType(), nullable = True),
        StructField("average_mpg", DoubleType(), nullable = True),
        StructField("total_fuel_gallons", DoubleType(), nullable = True),
        StructField("on_time_delivery_rate", DoubleType(), nullable = True),
        StructField("average_idle_hours", DoubleType(), nullable = True)
    ])
