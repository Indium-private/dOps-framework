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
    def __init__(self, data_directory):
        """Initializes the DataCrawler with the directory containing data files.
        
        Args:
            data_directory (str): Path to the directory containing data files.
        """
        self.data_directory = data_directory
        self.analyzer = AnalyzerEngine()  # Initialize Presidio ML PHI Detector

    def read_file(self, file_path):
        """Reads a file from the local filesystem."""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def classify_file(self, file_path):
        """Classifies the file format based on its extension or content."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if 'csv' in mime_type:
                return 'csv'
            elif 'json' in mime_type:
                return 'json'
            elif 'parquet' in mime_type:
                return 'parquet'
            elif 'avro' in mime_type:
                return 'avro'
        if file_path.endswith('.csv'):
            return 'csv'
        elif file_path.endswith('.json'):
            return 'json'
        elif file_path.endswith('.parquet'):
            return 'parquet'
        elif file_path.endswith('.avro'):
            return 'avro'
        return 'unknown'
    
    def detect_phi(self, column_name, sample_value):
        """Detects if a column contains PHI data using Presidio.
        
        Args:
            column_name (str): Column name to check.
            sample_value (str): Sample data from the column.
        
        Returns:
            bool: True if PHI is detected, False otherwise.
        """
        text_to_check = f"{column_name}: {sample_value}"  # Combine name + sample data
        results = self.analyzer.analyze(text=text_to_check, entities=None, language='en')

        return len(results) > 0  # If any PHI entity is detected, return True
    
    def infer_schema(self, file_path):
        """Infers schema from the file content and profiles the data."""
        file_type = self.classify_file(file_path)
        data = self.read_file(file_path)
        
        if file_type == 'csv':
            df = pd.read_csv(StringIO(data.decode('utf-8')))
        elif file_type == 'json':
            df = pd.DataFrame(json.loads(data.decode('utf-8')))
        elif file_type == 'parquet':
            df = pq.read_table(BytesIO(data)).to_pandas()
        elif file_type == 'avro':
            with BytesIO(data) as bio:
                reader = fastavro.reader(bio)
                df = pd.DataFrame([record for record in reader])
        else:
            return None
        
        # Convert dates properly
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass  # Keep as object if conversion fails

        schema = {
            "file_type": file_type,  # Include file type
            "columns": []
        }

        for col in df.columns:
            sample_value = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""  # Get sample value
            is_phi = self.detect_phi(col, sample_value)  # Run ML-based PHI detection

            schema["columns"].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'distinct_count': int(df[col].nunique()),
                'is_phi': is_phi,  # Add PHI flag
                'summary_statistics': self.compute_extended_statistics(df[col])
            })
        
        return schema
    
    def compute_extended_statistics(self, series):
        """Computes extended statistics for numeric and categorical data."""
        if pd.api.types.is_numeric_dtype(series):
            return {
                'min': series.min(),
                'q1': series.quantile(0.25),
                'median': series.median(),
                'q3': series.quantile(0.75),
                'max': series.max(),
                'mean': series.mean(),
                'std_dev': series.std(),
                'variance': series.var(),
                'skewness': series.skew(),
                'kurtosis': series.kurt(),
                'missing_percentage': series.isnull().mean() * 100,
                'outliers': self.detect_outliers(series)
            }
        else:
            return {
                'mode': series.mode().tolist(),
                'unique_count': series.nunique(),
                'missing_percentage': series.isnull().mean() * 100,
                'frequent_values': series.value_counts().head(5).to_dict()
            }
    
    def detect_outliers(self, series):
        """Detects potential outliers using the IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return series[(series < lower_bound) | (series > upper_bound)].tolist()
    
    # def save_to_yaml(self, file_name, schema):
    #     """Saves schema and profiling info to a YAML file."""
    #     with open(file_name, 'w') as f:
    #         yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    def save_to_json(self, file_name, schema):
        """Saves schema and profiling info to a JSON file.

        Args:
            file_name (str): Name of the output JSON file.
            schema (dict): Schema information to be saved.
        """
        json_file = file_name.replace('.', '_') + ".json"  # Ensure consistent naming

        # Custom function to handle unsupported types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):  # Convert Pandas timestamps to string
                return obj.isoformat()
            elif isinstance(obj, pd.Series):  # Convert Pandas Series to list
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):  # Convert DataFrame to dictionary
                return obj.to_dict(orient="records")
            elif isinstance(obj, dict):  # Ensure dictionary keys are strings
                return {str(k): convert_types(v) for k, v in obj.items()}
            return obj  # Return as is for other types

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=4, ensure_ascii=False, default=convert_types)

        print(f"Schema saved: {json_file}")

    def run(self):
        """Runs the crawler for a given local directory, inferring schemas and saving them to YAML files."""
        files = [os.path.join(self.data_directory, f) for f in os.listdir(self.data_directory)]
        
        for file in files:
            schema = self.infer_schema(file)
            if schema:
                output_file = os.path.basename(file).replace('.', '_') + '.yml'
                # self.save_to_yaml(output_file, schema)
                self.save_to_json(output_file, schema)
                print(f"Schema saved: {output_file}")

# Example usage
path = Path(r"D:\Indium Internal Work\Accelerators\crawler")
crawler = DataCrawler(data_directory = path)
crawler.run()
