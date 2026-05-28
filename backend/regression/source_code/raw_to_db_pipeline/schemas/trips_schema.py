from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, DateType
)

class TripsSchema:
    schema = StructType([
        StructField("trip_id", StringType(), nullable = False),
        StructField("load_id", StringType(), nullable = True),
        StructField("driver_id", StringType(), nullable = True),
        StructField("truck_id", StringType(), nullable = True),
        StructField("trailer_id", StringType(), nullable = True),
        StructField("dispatch_date", DateType(), nullable = True),
        StructField("actual_distance_miles", DoubleType(), nullable = True),
        StructField("actual_duration_hours", DoubleType(), nullable = True),
        StructField("fuel_gallons_used", DoubleType(), nullable = True),
        StructField("average_mpg", DoubleType(), nullable = True),
        StructField("idle_time_hours", DoubleType(), nullable = True),
        StructField("trip_status", StringType(), nullable = True)
    ])
