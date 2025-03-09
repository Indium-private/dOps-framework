from transformers import pipeline
from pathlib import Path
import pandas as pd
from presidio_analyzer import AnalyzerEngine


path = Path(r"D:\Indium Internal Work\Accelerators\testing datasets\large_customers.csv")
df = pd.read_csv(path, nrows=0)  # Read only the header row
field_names = [field.lower() for field in df.columns.tolist()]  # Convert to list
print(field_names)

analyzer = AnalyzerEngine()

def detect_pii(column_name):
    """Uses Presidio to detect general PII in column names."""
    results = analyzer.analyze(text = column_name, entities = analyzer.get_supported_entities(language="en"), language = "en")
    print(results)
    # return any(entity.entity_type in ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "PERSON", 'LOCATION', 'PHONE NUMBER', 'NRP', 'IBAN_CODE'] for entity in results)

for field in field_names:
    detect_pii(field)