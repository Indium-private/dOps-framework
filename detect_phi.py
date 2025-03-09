from transformers import pipeline
from pathlib import Path
import pandas as pd

path = Path(r"D:\Indium Internal Work\Accelerators\testing datasets\hospital_patient_records\payers.csv")
df = pd.read_csv(path, nrows=0)  # Read only the header row
field_names = df.columns.tolist()  # Convert to list
# print(field_names)

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")  # Load model once
# print(type(ner_pipeline))

def detect_phi(column_name, sensitive_domains={"PER", "LOC", "ORG", "MISC"}):
    """
    Uses a Transformer-based NER model to detect if a column name suggests PHI/PII.
    """
    results = ner_pipeline(column_name.replace("_", " "))
    print(results)

    for entity in results:
        entity_label = entity.get('entity_group') or entity.get('entity')
        if entity_label in sensitive_domains:
            print(f"{entity} belongs to {entity_label}")
    
    return False

for column in field_names:
    detect_phi(column_name=column)



from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
supported_entities = analyzer.get_supported_entities(language="en")
print(supported_entities)