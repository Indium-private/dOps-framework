import chardet
import pandas as pd
from pathlib import Path

def detect_file_encoding(file_path, sample_size = 10000, confidence_threshold = 1):
    """
    Detects the encoding of a file by reading a sample of its content.

    Args:
        file_path (str or Path): Path to the file.
        sample_size (int): Number of bytes to read for detection (default is 10,000).
        confidence_threshold (float): Minimum confidence for detected encoding.

    Returns:
        str: Detected encoding or 'utf-8' as a fallback.
    """
    file_path = Path(file_path)  # Ensure it's a Path object

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("rb") as f:
        raw_data = f.read(sample_size)  # Read a sample of the file

    result = chardet.detect(raw_data)  # Detect encoding
    encoding = result.get("encoding")
    confidence = result.get("confidence", 0)

    # Fallback if confidence is too low or encoding is None
    if not encoding or confidence < confidence_threshold:
        return "utf-8"

    return encoding

# Example usage:
file_path = Path(r"D:\Indium Internal Work\Accelerators\testing datasets\global_electronics_retailer\Customers.csv")
detected_encoding = detect_file_encoding(file_path)
print(f"Detected Encoding: {detected_encoding}")

# Now, read the file safely with detected encoding
try:
    df = pd.read_csv(file_path, encoding=detected_encoding, encoding_errors="replace")  # "replace" handles decoding errors
    print(df.head())  # Print first few rows
except Exception as e:
    print(f"Error reading file: {e}")
