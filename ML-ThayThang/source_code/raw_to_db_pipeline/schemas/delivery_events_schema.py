from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, BooleanType, TimestampType
)

class DeliveryEventsSchema:
    schema = StructType([
        StructField("event_id", StringType(), nullable = False),
        StructField("load_id", StringType(), nullable = True),
        StructField("trip_id", StringType(), nullable = True),
        StructField("event_type", StringType(), nullable = True),
        StructField("facility_id", StringType(), nullable = True),
        StructField("scheduled_datetime", TimestampType(), nullable = True),
        StructField("actual_datetime", TimestampType(), nullable = True),
        StructField("detention_minutes", IntegerType(), nullable = True),
        StructField("on_time_flag", BooleanType(), nullable = True),
        StructField("location_city", StringType(), nullable = True),
        StructField("location_state", StringType(), nullable = True)
    ])
