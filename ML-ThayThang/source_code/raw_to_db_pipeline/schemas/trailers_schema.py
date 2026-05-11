from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DateType
)

class TrailersSchema:
    schema = StructType([
        StructField("trailer_id", StringType(), nullable = False),
        StructField("trailer_number", IntegerType(), nullable = True),
        StructField("trailer_type", StringType(), nullable = True),
        StructField("length_feet", IntegerType(), nullable = True),
        StructField("model_year", IntegerType(), nullable = True),
        StructField("vin", StringType(), nullable = True),
        StructField("acquisition_date", DateType(), nullable = True),
        StructField("status", StringType(), nullable = True),
        StructField("current_location", StringType(), nullable = True)
    ])
