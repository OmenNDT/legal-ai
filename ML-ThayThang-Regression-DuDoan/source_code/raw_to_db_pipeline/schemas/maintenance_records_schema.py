from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, DateType
)

class MaintenanceRecordsSchema:
    schema = StructType([
        StructField("maintenance_id", StringType(), nullable = False),
        StructField("truck_id", StringType(), nullable = True),
        StructField("maintenance_date", DateType(), nullable = True),
        StructField("maintenance_type", StringType(), nullable = True),
        StructField("odometer_reading", DoubleType(), nullable = True),
        StructField("labor_hours", DoubleType(), nullable = True),
        StructField("labor_cost", DoubleType(), nullable = True),
        StructField("parts_cost", DoubleType(), nullable = True),
        StructField("total_cost", DoubleType(), nullable = True),
        StructField("facility_location", StringType(), nullable = True),
        StructField("downtime_hours", DoubleType(), nullable = True),
        StructField("service_description", StringType(), nullable = True)
    ])
