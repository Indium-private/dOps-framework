from transformers import pipeline
from pathlib import Path
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from sentence_transformers import SentenceTransformer, util


path = Path(r"D:\Indium Internal Work\Accelerators\testing datasets\large_customers.csv")
df = pd.read_csv(path, nrows=0)  # Read only the header row
field_names = [field.lower() for field in df.columns.tolist()]  # Convert to list
print(field_names)


class PIIDetector:
    def __init__(self, pii_entities=None, similarity_threshold=0.7):
        """
        Initializes the PII Detector with a predefined list of PII entities.
        
        Args:
            pii_entities (list, optional): List of known PII entity names.
            similarity_threshold (float, optional): Threshold for similarity detection.
        """
        self.similarity_threshold = similarity_threshold

        # Default PII entities if none are provided
        if pii_entities is None:
            self.pii_entities = [
                "email", "phone number", "credit card", "ssn", "social security number", 
                "name", "address", "dob", "date of birth", "passport number", "bank account"
            ]
        else:
            self.pii_entities = pii_entities

        # Load sentence transformer model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def detect_pii(self, column_name):
        """
        Uses Sentence Transformers to detect PII in column names.
        
        Args:
            column_name (str): The column name to check.

        Returns:
            bool: True if column name is considered PII.
        """
        column_embedding = self.model.encode(column_name, convert_to_tensor=True)
        pii_embeddings = self.model.encode(self.pii_entities, convert_to_tensor=True)

        # Compute cosine similarity between column and PII entities
        similarities = util.cos_sim(column_embedding, pii_embeddings)

        # If any similarity score is above the threshold, flag as PII
        return any(similarity > self.similarity_threshold for similarity in similarities[0])

# Example Usage
detector = PIIDetector()

for column in field_names:
    print(detector.detect_pii(column))