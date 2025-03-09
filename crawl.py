import os
import yaml
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
import fastavro
import mimetypes
from io import StringIO, BytesIO
from pathlib import Path
from presidio_analyzer import AnalyzerEngine

class DataCrawler:
    def __init__(self, file_path):
        """Initializes the DataCrawler with the path to a single data file.
        
        Args:
            file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.analyzer = AnalyzerEngine()  # Initialize Presidio PHI Detector

    def read_file(self):
        """Reads the specified file from the local filesystem."""
        with open(self.file_path, 'rb') as f:
            return f.read()
        
    def detect_phi(self, column_name):
        """Uses Presidio to detect PHI/PII in column names."""
        results = self.analyzer.analyze(text=column_name, entities = self.analyzer.get_supported_entities(language="en"), language="en")
        return len(results) > 0  # Returns True if PHI is detected

    def infer_schema(self):
        """
        Infers schema from the file content and profiles the data.
        """
        file_ext = Path(self.file_path).suffix.lower()
        
        with open(self.file_path, "rb") as f:
            data = f.read()

        if file_ext == '.csv':
            df = pd.read_csv(StringIO(data.decode('utf-8')))
        elif file_ext == '.json':
            df = pd.DataFrame(json.loads(data.decode('utf-8')))
        elif file_ext == '.parquet':
            df = pq.read_table(BytesIO(data)).to_pandas()
        elif file_ext == '.avro':
            with BytesIO(data) as bio:
                reader = fastavro.reader(bio)
                df = pd.DataFrame([record for record in reader])
        else:
            print("Unsupported file type.")
            return None

        schema = {
            "file_type": file_ext,
            "columns": []
        }

        for col in df.columns:
            is_phi = self.detect_phi(col)
            schema["columns"].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'total_count': int(len(df[col])),
                'distinct_count': int(df[col].nunique()),
                'is_phi': is_phi
            })

        return schema    

    def save_to_json(self, schema):
        """Saves schema and profiling info to a JSON file."""
        json_file = os.path.basename(self.file_path).replace('.', '_') + ".json"

        def convert_types(obj):
            """Handles serialization of NumPy and Pandas types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            elif isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            return obj

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=4, ensure_ascii=False, default=convert_types)

        print(f"Schema saved: {json_file}")

    def run(self):
        """
        Runs the crawler for the given file.
        """
        schema = self.infer_schema()
        if schema:
            self.save_to_json(schema)

# Example usage
file_path = r"D:\Indium Internal Work\Accelerators\testing datasets\global_electronics_retailer\Customers.csv"
crawler = DataCrawler(file_path=file_path)
crawler.run()