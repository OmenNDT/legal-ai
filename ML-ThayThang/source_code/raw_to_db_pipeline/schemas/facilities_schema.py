from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)

class FacilitiesSchema:
    schema = StructType([
        StructField("facility_id", StringType(), nullable = False),
        StructField("facility_name", StringType(), nullable = True),
        StructField("facility_type", StringType(), nullable = True),
        StructField("city", StringType(), nullable = True),
        StructField("state", StringType(), nullable = True),
        StructField("latitude", DoubleType(), nullable = True),
        StructField("longitude", DoubleType(), nullable = True),
        StructField("dock_doors", IntegerType(), nullable = True),
        StructField("operating_hours", StringType(), nullable = True)
    ])
