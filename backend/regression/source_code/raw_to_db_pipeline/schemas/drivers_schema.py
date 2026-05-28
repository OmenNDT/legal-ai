from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DateType
)

class DriversSchema:
    schema = StructType([
        StructField("driver_id", StringType(), nullable = False),
        StructField("first_name", StringType(), nullable = True),
        StructField("last_name", StringType(), nullable = True),
        StructField("hire_date", DateType(), nullable = True),
        StructField("termination_date", DateType(), nullable = True),
        StructField("license_number", StringType(), nullable = True),
        StructField("license_state", StringType(), nullable = True),
        StructField("date_of_birth", DateType(), nullable = True),
        StructField("home_terminal", StringType(), nullable = True),
        StructField("employment_status", StringType(), nullable = True),
        StructField("cdl_class", StringType(), nullable = True),
        StructField("years_experience", IntegerType(), nullable = True)
    ])
