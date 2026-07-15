import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import (
    FILTERED_DATA_PATH, VECTOR_STORE_PATH, 
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, SAMPLE_SIZE
)

def run_task2():
    print("📂 Loading filtered data...")
    df = pd.read_csv(FILTERED_DATA_PATH)
    
    # ---- Stratified Sampling ----
    print(f"📊 Creating stratified sample of ~{SAMPLE_SIZE} rows...")
    # Calculate proportional sample per product
    sample_per_product = {}
    for product in df['product'].unique():
        product_count = len(df[df['product'] == product])
        sample_per_product[product] = int(SAMPLE_SIZE * (product_count / len(df)))
    
    # Sample with replacement if needed (to ensure we hit target)
    df_sample = pd.DataFrame()
    for product, n in sample_per_product.items():
        product_df = df[df['product'] == product]
        sampled = product_df.sample(n=min(n, len(product_df)), random_state=42)
        df_sample = pd.concat([df_sample, sampled])
    
    # Adjust if we need exactly SAMPLE_SIZE
    if len(df_sample) < SAMPLE_SIZE:
        remaining = SAMPLE_SIZE - len(df_sample)
        extra = df.drop(df_sample.index).sample(n=min(remaining, len(df)-len(df_sample)), random_state=42)
        df_sample = pd.concat([df_sample, extra])
    
    print(f"✅ Sampled {len(df_sample)} complaints. Product distribution:\n{df_sample['product'].value_counts()}")
    
    # ---- Chunking ----
    print(f"✂️ Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_data = []
    for idx, row in df_sample.iterrows():
        text = row['clean_narrative']
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                'complaint_id': row.get('complaint_id', idx),
                'product': row['product'],
                'chunk_index': i,
                'text': chunk
            })
    
    df_chunks = pd.DataFrame(chunks_data)
    print(f"🧩 Generated {len(df_chunks)} chunks.")
    
    # ---- Embedding ----
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("⚡ Generating embeddings...")
    embeddings = model.encode(df_chunks['text'].tolist(), show_progress_bar=True)
    
    # ---- ChromaDB Store ----
    print(f"💾 Persisting vector store to {VECTOR_STORE_PATH}...")
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # Delete existing collection if it exists to start fresh
    try:
        client.delete_collection("complaints")
    except:
        pass
    
    collection = client.create_collection(name="complaints")
    
    # Prepare data
    ids = [f"comp_{row['complaint_id']}_chunk_{row['chunk_index']}" for _, row in df_chunks.iterrows()]
    metadatas = df_chunks[['complaint_id', 'product', 'chunk_index']].to_dict('records')
    documents = df_chunks['text'].tolist()
    
    # Add in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        collection.add(
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    
    print(f"✅ Vector store ready with {collection.count()} chunks.")
    return collection, model

if name == "main":
    run_task2()
