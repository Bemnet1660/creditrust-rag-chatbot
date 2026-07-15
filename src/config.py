import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(file)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")
FILTERED_DATA_PATH = os.path.join(BASE_DIR, "data", "filtered_complaints.csv")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

# Filtering
PRODUCTS = ['Credit card', 'Personal loan', 'Savings account', 'Money transfer']

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Sampling
SAMPLE_SIZE = 12000  # ~10k-15k

# RAG
TOP_K = 5
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # or "HuggingFaceH4/zephyr-7b-beta"
