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
from sentence_transformers import SentenceTransformer, util

class DataCrawler:
    def __init__(self, file_path, output_dir, similarity_threshold = 0.7):
        """Initializes the DataCrawler with paths.
        
        Args:
            file_path (str): Path to the data file.
            output_dir (str): Path to the output directory for JSON.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.analyzer = AnalyzerEngine()  # Initialize Presidio PHI & PII Detector
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold
        self.pii_entities = [
                "email", "phone number", "credit card", "ssn", "social security number", 
                "name", "address", "dob", "date of birth", "passport number", "bank account"
            ]

    def read_file(self):
        """Reads the specified file from the local filesystem."""
        with open(self.file_path, 'rb') as f:
            return f.read()
  
    def detect_phi(self, column_name):
        """Uses Presidio to detect PHI in column names."""
        results = self.analyzer.analyze(text=column_name, entities=self.analyzer.get_supported_entities(language="en"), language="en")
        return any(entity.entity_type in ["MEDICAL_CONDITION", "MEDICAL_TREATMENT", "US_SSN"] for entity in results)

    def detect_pii(self, column_name):
        column_embedding = self.model.encode(column_name, convert_to_tensor=True)
        pii_embeddings = self.model.encode(self.pii_entities, convert_to_tensor=True)

        similarities = util.cos_sim(column_embedding, pii_embeddings)

        # If any similarity score is above the threshold, flag as PII
        return any(similarity > self.similarity_threshold for similarity in similarities[0])

    def infer_schema(self):
        """Infers schema from the file content and profiles the data."""
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

        for col in df.columns: # converting the field names to lower case to reduce noise
            is_phi = self.detect_phi(col)
            is_pii = self.detect_pii(col)
            schema["columns"].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'total_count': int(len(df[col])),
                'distinct_count': int(df[col].nunique()),
                'is_phi': is_phi,
                'is_pii': is_pii
            })

        return schema    

    def save_to_json(self, schema):
        """Saves schema and profiling info to a JSON file inside the specified output directory."""
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Construct the JSON file name (same as input file but with .json extension)
        json_file_name = os.path.basename(self.file_path).replace('.', '_') + ".json"
        json_file_path = os.path.join(self.output_dir, json_file_name)

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

        # Save the schema to the specified directory
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=4, ensure_ascii=False, default=convert_types)

        print(f"Schema saved: {json_file_path}")

    def run(self):
        """Runs the crawler for the given file."""
        schema = self.infer_schema()
        if schema:
            self.save_to_json(schema)

# Example usage
file_path = r"D:\Indium Internal Work\Accelerators\testing datasets\crm_sales_opportunities\sales_teams.csv"
output_dir = r"D:\Indium Internal Work\Accelerators\inferred_schemas"

crawler = DataCrawler(file_path=file_path, output_dir=output_dir)
crawler.run()
