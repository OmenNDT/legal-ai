import argparse
import os
import re
from functools import reduce
from pathlib import Path
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

class TextCleaner:

    _WHITESPACE_RE = re.compile(r"\s+")
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return text
        text = TextCleaner._WHITESPACE_RE.sub(" ", text).strip()
        text = TextCleaner._CONTROL_RE.sub("", text)
        return text

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split()) if text else 0
    clean_udf = F.udf(clean.__func__, StringType())
    word_count_udf = F.udf(word_count.__func__, IntegerType())

class PostgresConfig:

    def __init__(self, host: str, port: int, db: str, user: str, password: str, table: str = "qa_data") -> None:
        self.url = f"jdbc:postgresql://{host}:{port}/{db}"
        self.table = table
        self.user = user
        self.password = password
        self.driver = "org.postgresql.Driver"

    def jdbc_options(self) -> dict:
        return {
            "url": self.url,
            "dbtable": self.table,
            "user": self.user,
            "password": self.password,
            "driver": self.driver
        }

class LegalDataPreprocessor:

    TEXT_COLS = ("Question", "Answer")

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    def read(self, path: str) -> DataFrame:
        df = self.spark.read.parquet(path)
        df.printSchema()
        return df

    def clean_text(self, df: DataFrame) -> DataFrame:
        for col in self.TEXT_COLS:
            df = df.withColumn(col, TextCleaner.clean_udf(F.col(col)))
        return df

    def drop_empty(self, df: DataFrame) -> DataFrame:
        conditions = [
            F.col(col).isNotNull() & (F.length(F.col(col)) > 0)
            for col in self.TEXT_COLS
        ]
        return df.filter(reduce(lambda a, b: a & b, conditions))

    def filter_legal_answers(self, df: DataFrame) -> DataFrame:
        prefixes = (
            "Căn cứ", "Theo", "Tại", "Từ",
            "Khoản", "Điều", "Điểm", "Tiểu", "Tiết", "Mục",
            "Quy định", "Quy trình", "Quy chuẩn", "Quyền", "Quyết định"
        )
        condition = reduce(
            lambda a, b: a | b,
            [F.col("Answer").startswith(p) for p in prefixes]
        )
        excluded_prefixes = ("Quyền của nhà", "Quyền sở hữu trí tuệ")
        excluded = reduce(
            lambda a, b: a | b,
            [F.col("Answer").startswith(p) for p in excluded_prefixes]
        )
        return df.filter(condition & ~excluded)

    def write_postgres(self, df: DataFrame, pg: PostgresConfig) -> None:
        df = (
            df.select(
                F.col("ID").alias("id"),
                F.col("ID1").alias("id1"),
                F.col("Question").alias("question"),
                F.col("Answer").alias("answer"),
                F.col("idx").alias("idx")
            )
        )
        (
            df.write
            .format("jdbc")
            .options(**pg.jdbc_options())
            .mode("append")
            .save()
        )
        print(f"[INFO] Đã ghi xong vào {pg.url} bảng [{pg.table}]")

    def run(self, input_path: str, pg: PostgresConfig) -> None:
        df = self.read(input_path)
        df = self.clean_text(df)
        df = self.drop_empty(df)
        df = self.filter_legal_answers(df)
        self.write_postgres(df, pg)

class SparkJobRunner:

    APP_NAME = "legal-ai-preprocess"
    JDBC_JAR_LOCAL = str(Path(__file__).resolve().parents[3] / "driver" / "postgresql-42.7.7.jar")
    JDBC_JAR_HDFS = "hdfs://master:9000/legal-ai/driver/postgresql-42.7.7.jar"

    def __init__(self, input_path: str, pg: PostgresConfig) -> None:
        self.input_path = input_path
        self.pg = pg

    def _build_spark(self) -> SparkSession:
        return (
            SparkSession.builder
            .appName(self.APP_NAME)
            .config("spark.jars", self.JDBC_JAR_HDFS)
            .config("spark.driver.extraClassPath", self.JDBC_JAR_LOCAL)
            .config("spark.sql.parquet.enableVectorizedReader", "true")
            .config("spark.io.compression.codec", "lz4")
            .config("spark.shuffle.compress", "false")
            .config("spark.shuffle.spill.compress", "false")
            .getOrCreate()
        )

    def execute(self) -> None:
        spark = self._build_spark()
        spark.sparkContext.setLogLevel("WARN")
        try:
            LegalDataPreprocessor(spark).run(self.input_path, self.pg)
        finally:
            spark.stop()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Legal-AI Spark → PostgreSQL job")
    parser.add_argument("--input", default = "hdfs://master:9000/legal-ai/data/data.parquet", help = "Parquet đầu vào")
    parser.add_argument("--pg-table", default = "qa_data", help = "Tên bảng đích")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pg = PostgresConfig(
        host = os.environ["POSTGRES_HOST"],
        port = int(os.environ["POSTGRES_PORT"]),
        db = os.environ["POSTGRES_DB"],
        user = os.environ["POSTGRES_USER"],
        password = os.environ["POSTGRES_PASSWORD"],
        table = args.pg_table
    )
    SparkJobRunner(args.input, pg).execute()

# SELECT COUNT(*) FROM train
# WHERE question IS NOT NULL AND answer IS NOT NULL
#   AND LENGTH(question) > 0 AND LENGTH(answer) > 0
#   AND (
#     answer LIKE 'Căn cứ%' OR answer LIKE 'Theo%' OR answer LIKE 'Tại%' OR answer LIKE 'Từ%'
#     OR answer LIKE 'Khoản%' OR answer LIKE 'Điều%' OR answer LIKE 'Điểm%' OR answer LIKE 'Tiểu%'
#     OR answer LIKE 'Tiết%'  OR answer LIKE 'Mục%'
#     OR answer LIKE 'Quy định%' OR answer LIKE 'Quy trình%' OR answer LIKE 'Quy chuẩn%'
#     OR answer LIKE 'Quyền%' OR answer LIKE 'Quyết định%'
#   )
#   AND answer NOT LIKE 'Quyền của nhà%'
#   AND answer NOT LIKE 'Quyền sở hữu trí tuệ%';