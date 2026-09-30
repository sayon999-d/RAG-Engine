import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_KEY")
print(f"Key loaded: {API_KEY[:10]}..." if API_KEY else "ERROR: No key found!")

client = InferenceClient(token=API_KEY)

print("\nTesting embedding...")
try:
    result = client.feature_extraction(
        "Hello world, this is a test.", model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"Success! Shape: {result.shape}")
    if len(result.shape) == 1:
        preview = result[:5]
    else:
        preview = result[0][:5]
    print(f"First 5 values: {preview}")
    print("\nHuggingFace API is working!")
except Exception as e:
    print(f"Error: {e}")
