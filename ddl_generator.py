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
    def __init__(self, schema_dir, database_name):
        """Initializes the DDL Generator with schema directory and database name.
        
        Args:
            schema_dir (str): Directory containing JSON schema files.
            database_name (str): Target database name.
        """
        self.schema_dir = Path(schema_dir)
        self.database_name = database_name

    def load_schema(self, schema_file):
        """Loads a JSON schema file.
        
        Args:
            schema_file (Path): Path to the schema JSON file.
        
        Returns:
            dict: Parsed schema dictionary.
        """
        with schema_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def map_dtype(self, dtype):
        """Maps Pandas/PySpark data types to ANSI SQL types."""
        return DATA_TYPE_MAPPING.get(dtype, "TEXT")  # Default to TEXT for unknown types

    def generate_ddl(self, schema):
        """Generates a CREATE TABLE IF NOT EXISTS statement based on the schema.

        Args:
            schema (dict): Parsed schema dictionary.

        Returns:
            str: ANSI SQL-compliant CREATE TABLE IF NOT EXISTS statement.
        """
        table_name = schema["file_type"] + "_table"  # Default table name
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

    def save_ddl(self, ddl, output_file):
        """Saves the generated DDL to a file."""
        with output_file.open("w", encoding="utf-8") as f:
            f.write(ddl + "\n")
        print(f"DDL saved: {output_file}")

    def run(self):
        """Processes all schema JSON files and generates DDL statements."""
        schema_files = list(self.schema_dir.glob("*.json"))

        for schema_file in schema_files:
            schema = self.load_schema(schema_file)
            ddl_statement = self.generate_ddl(schema)

            output_ddl_file = schema_file.with_suffix(".sql")
            self.save_ddl(ddl_statement, output_ddl_file)

# Example usage
schema_directory = Path(r"D:\Indium Internal Work\Accelerators\inferred_schemas")
database_name = "my_database"
ddl_generator = DDLGenerator(schema_directory, database_name)
ddl_generator.run()
