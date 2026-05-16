from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, DateType
)

class TrucksSchema:
    schema = StructType([
        StructField("truck_id", StringType(), nullable = False),
        StructField("unit_number", IntegerType(), nullable = True),
        StructField("make", StringType(), nullable = True),
        StructField("model_year", IntegerType(), nullable = True),
        StructField("vin", StringType(), nullable = True),
        StructField("acquisition_date", DateType(), nullable = True),
        StructField("acquisition_mileage", DoubleType(), nullable = True),
        StructField("fuel_type", StringType(), nullable = True),
        StructField("tank_capacity_gallons", DoubleType(), nullable = True),
        StructField("status", StringType(), nullable = True),
        StructField("home_terminal", StringType(), nullable = True)
    ])
