from raw_to_db_pipeline.csv_to_postgres_pipeline import CsvToPostgresPipeline

if __name__ == "__main__":
    pipeline = CsvToPostgresPipeline()
    pipeline.run()