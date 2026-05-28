import yaml
import psycopg2
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

from .spark_session_builder import SparkSessionBuilder
from .schemas import (
    CustomersSchema,
    DeliveryEventsSchema,
    DriverMonthlyMetricsSchema,
    DriversSchema,
    FacilitiesSchema,
    FuelPurchasesSchema,
    LoadsSchema,
    MaintenanceRecordsSchema,
    RoutesSchema,
    SafetyIncidentsSchema,
    TrailersSchema,
    TripsSchema,
    TruckUtilizationMetricsSchema,
    TrucksSchema
)

TABLE_CONFIG: list[dict] = [
    {"csv": "drivers.csv", "table": "drivers", "schema": DriversSchema.schema, "pk": ["driver_id"]},
    {"csv": "trucks.csv", "table": "trucks", "schema": TrucksSchema.schema, "pk": ["truck_id"]},
    {"csv": "trailers.csv", "table": "trailers", "schema": TrailersSchema.schema, "pk": ["trailer_id"]},
    {"csv": "customers.csv", "table": "customers", "schema": CustomersSchema.schema, "pk": ["customer_id"]},
    {"csv": "facilities.csv", "table": "facilities", "schema": FacilitiesSchema.schema, "pk": ["facility_id"]},
    {"csv": "routes.csv", "table": "routes", "schema": RoutesSchema.schema, "pk": ["route_id"]},
    {"csv": "loads.csv", "table": "loads", "schema": LoadsSchema.schema, "pk": ["load_id"]},
    {"csv": "trips.csv", "table": "trips", "schema": TripsSchema.schema, "pk": ["trip_id"]},
    {"csv": "fuel_purchases.csv", "table": "fuel_purchases", "schema": FuelPurchasesSchema.schema, "pk": ["fuel_purchase_id"]},
    {"csv": "maintenance_records.csv", "table": "maintenance_records", "schema": MaintenanceRecordsSchema.schema, "pk": ["maintenance_id"]},
    {"csv": "delivery_events.csv", "table": "delivery_events", "schema": DeliveryEventsSchema.schema, "pk": ["event_id"]},
    {"csv": "safety_incidents.csv", "table": "safety_incidents", "schema": SafetyIncidentsSchema.schema, "pk": ["incident_id"]},
    {"csv": "driver_monthly_metrics.csv", "table": "driver_monthly_metrics", "schema": DriverMonthlyMetricsSchema.schema, "pk": ["driver_id", "month"]},
    {"csv": "truck_utilization_metrics.csv", "table": "truck_utilization_metrics", "schema": TruckUtilizationMetricsSchema.schema, "pk": ["truck_id", "month"]}
]

class CsvToPostgresPipeline:

    def __init__(self, config_path: str | Path | None = None, spark_mode: str = "auto"):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with Path(config_path).open() as f:
            self._config = yaml.safe_load(f)

        self._data_raw_dir = Path(__file__).parent.parent / "data_raw"
        self._spark_builder = SparkSessionBuilder(config_path = config_path, mode = spark_mode)

    def _jdbc_url(self) -> str:
        db = self._config["database"]
        return f"jdbc:postgresql://{db['host']}:{db['port']}/{db['database_name']}"

    def _jdbc_props(self) -> dict:
        db = self._config["database"]
        return {
            "user": db["username"],
            "password": db["password"],
            "driver": db["driver"]
        }

    def _psycopg2_conn(self):
        db = self._config["database"]
        return psycopg2.connect(
            host = db["host"],
            port = db["port"],
            dbname = db["database_name"],
            user = db["username"],
            password = db["password"]
        )

    def _read_csv(self, spark: SparkSession, csv_filename: str, schema: StructType) -> DataFrame:
        csv_path = str(self._data_raw_dir / csv_filename)
        return (
            spark.read
            .option("header", "true")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .option("dateFormat", "yyyy-MM-dd")
            .schema(schema)
            .csv(csv_path)
        )

    def _build_upsert_sql(self, table: str, columns: list[str], pk: list[str]) -> str:
        tmp = f"_tmp_{table}"
        non_pk = [c for c in columns if c not in pk]
        conflict_cols = ", ".join(pk)
        update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk)
        return (
            f"INSERT INTO {table} SELECT * FROM {tmp} "
            f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_set}"
        )

    def _upsert_table(self, df: DataFrame, table: str, pk: list[str], jdbc_url: str, jdbc_props: dict) -> None:
        tmp = f"_tmp_{table}"
        columns = [f.name for f in df.schema.fields]
        df.write.jdbc(url = jdbc_url, table = tmp, mode = "overwrite", properties = jdbc_props)
        upsert_sql = self._build_upsert_sql(table, columns, pk)
        with self._psycopg2_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(upsert_sql)
                cur.execute(f"DROP TABLE IF EXISTS {tmp}")
            conn.commit()

    def run(self) -> None:
        jdbc_url = self._jdbc_url()
        jdbc_props = self._jdbc_props()

        with self._spark_builder as spark:
            for entry in TABLE_CONFIG:
                print(f"[pipeline] Loading {entry['csv']} → {entry['table']} ...")
                df = self._read_csv(spark, entry["csv"], entry["schema"])
                self._upsert_table(df, entry["table"], entry["pk"], jdbc_url, jdbc_props)
                print(f"[pipeline] Upserted: {entry['table']} ({df.count()} rows)")

        print("[pipeline] All tables upserted successfully.")