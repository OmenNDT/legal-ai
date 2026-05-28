from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType
)

class TruckUtilizationMetricsSchema:
    schema = StructType([
        StructField("truck_id", StringType(), nullable = False),
        StructField("month", DateType(), nullable = False),
        StructField("trips_completed", IntegerType(), nullable = True),
        StructField("total_miles", DoubleType(), nullable = True),
        StructField("total_revenue", DoubleType(), nullable = True),
        StructField("average_mpg", DoubleType(), nullable = True),
        StructField("maintenance_events", IntegerType(), nullable = True),
        StructField("maintenance_cost", DoubleType(), nullable = True),
        StructField("downtime_hours", DoubleType(), nullable = True),
        StructField("utilization_rate", DoubleType(), nullable = True)
    ])
