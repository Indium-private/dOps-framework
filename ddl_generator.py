import json
from pathlib import Path

# Define a mapping from Pandas/PySpark types to ANSI SQL types
DATA_TYPE_MAPPING = {
    "int64": "BIGINT",
    "int32": "INTEGER",
    "float64": "DOUBLE PRECISION",
    "float32": "FLOAT",
    "object": "TEXT",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
}

class DDLGenerator:
    def __init__(self, schema_path, database_name, output_dir="generated_ddl"):
        """
        Initializes the DDL Generator with schema file/directory, database name, and output directory.
        
        Args:
            schema_path (str or Path): Path to a JSON file or directory containing schema files.
            database_name (str): Target database name.
            output_dir (str or Path): Directory to save generated DDL files.
        """
        self.schema_path = Path(schema_path)
        self.database_name = database_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # ensure the folder exists

    def load_schema(self, schema_file):
        with schema_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def map_dtype(self, dtype):
        return DATA_TYPE_MAPPING.get(dtype, "TEXT")

    def generate_ddl(self, schema):
        table_name = schema["file_type"] + "_table"
        columns_definitions = []

        for column in schema["columns"]:
            col_name = column["name"]
            col_type = self.map_dtype(column["dtype"])
            is_nullable = "NULL" if column["null_count"] > 0 else "NOT NULL"
            columns_definitions.append(f"    {col_name} {col_type} {is_nullable}")

        columns_sql = ",\n".join(columns_definitions)

        ddl = f"""
CREATE TABLE IF NOT EXISTS {self.database_name}.{table_name} (
{columns_sql}
);
""".strip()
        return ddl

    def save_ddl(self, ddl, schema_file):
        output_file_name = schema_file.stem + ".sql"
        output_file_path = self.output_dir / output_file_name
        with output_file_path.open("w", encoding="utf-8") as f:
            f.write(ddl + "\n")
        print(f"DDL saved: {output_file_path}")

    def run(self):
        if self.schema_path.is_file():
            schema_files = [self.schema_path]
        elif self.schema_path.is_dir():
            schema_files = list(self.schema_path.glob("*.json"))
        else:
            print("Invalid schema path provided.")
            return

        for schema_file in schema_files:
            schema = self.load_schema(schema_file)
            ddl = self.generate_ddl(schema)
            self.save_ddl(ddl, schema_file)

schema_input = r"inferred_schemas\admin_staff_branch_csv.json"
database_name = "my_database"
ddl_generator = DDLGenerator(schema_input, database_name)
ddl_generator.run()
