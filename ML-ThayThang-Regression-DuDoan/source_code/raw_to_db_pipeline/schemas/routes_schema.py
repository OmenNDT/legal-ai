from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)

class RoutesSchema:
    schema = StructType([
        StructField("route_id", StringType(), nullable = False),
        StructField("origin_city", StringType(), nullable = True),
        StructField("origin_state", StringType(), nullable = True),
        StructField("destination_city", StringType(), nullable = True),
        StructField("destination_state", StringType(), nullable = True),
        StructField("typical_distance_miles", DoubleType(), nullable = True),
        StructField("base_rate_per_mile", DoubleType(), nullable = True),
        StructField("fuel_surcharge_rate", DoubleType(), nullable = True),
        StructField("typical_transit_days", IntegerType(), nullable = True)
    ])
