from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType, TimestampType
)

class SafetyIncidentsSchema:
    schema = StructType([
        StructField("incident_id", StringType(), nullable = False),
        StructField("trip_id", StringType(), nullable = True),
        StructField("truck_id", StringType(), nullable = True),
        StructField("driver_id", StringType(), nullable = True),
        StructField("incident_date", TimestampType(), nullable = True),
        StructField("incident_type", StringType(), nullable = True),
        StructField("location_city", StringType(), nullable = True),
        StructField("location_state", StringType(), nullable = True),
        StructField("at_fault_flag", BooleanType(), nullable = True),
        StructField("injury_flag", BooleanType(), nullable = True),
        StructField("vehicle_damage_cost", DoubleType(), nullable = True),
        StructField("cargo_damage_cost", DoubleType(), nullable = True),
        StructField("claim_amount", DoubleType(), nullable = True),
        StructField("preventable_flag", BooleanType(), nullable = True),
        StructField("description", StringType(), nullable = True)
    ])
