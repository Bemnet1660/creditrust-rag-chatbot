import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RAW_DATA_PATH, FILTERED_DATA_PATH, PRODUCTS
from src.utils import clean_complaint_text

def run_task1():
    print("📂 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # ---- EDA ----
    print("\n📊 Initial shape:", df.shape)
    print("\nProduct distribution:\n", df['product'].value_counts())
    
    # Narrative length analysis
    df['narrative_len'] = df['consumer_complaint_narrative'].fillna('').apply(lambda x: len(str(x).split()))
    print(f"\n📏 Narrative length stats: min={df['narrative_len'].min()}, max={df['narrative_len'].max()}, mean={df['narrative_len'].mean():.1f}")
    
    # Plot distribution (save to file)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    df['product'].value_counts().plot(kind='bar', title='Products')
    plt.subplot(1,2,2)
    df['narrative_len'].hist(bins=50, title='Narrative Length (words)')
    plt.tight_layout()
    plt.savefig('eda_plots.png')
    print("📈 EDA plots saved as eda_plots.png")
    
    # ---- Filtering ----
    print(f"\n🔍 Filtering to products: {PRODUCTS}")
    df_filtered = df[df['product'].isin(PRODUCTS)].copy()
    
    # Drop empty narratives
    df_filtered = df_filtered[df_filtered['consumer_complaint_narrative'].notna()]
    df_filtered = df_filtered[df_filtered['consumer_complaint_narrative'].str.strip() != '']
    print(f"✅ After filtering: {len(df_filtered)} rows")
    
    # ---- Cleaning ----
    print("\n🧹 Cleaning text narratives...")
    df_filtered['clean_narrative'] = df_filtered['consumer_complaint_narrative'].apply(clean_complaint_text)
    
    # Drop rows that became empty after cleaning
    df_filtered = df_filtered[df_filtered['clean_narrative'].str.strip() != '']
    
    # ---- Save ----
    os.makedirs(os.path.dirname(FILTERED_DATA_PATH), exist_ok=True)
    df_filtered.to_csv(FILTERED_DATA_PATH, index=False)
    print(f"💾 Cleaned data saved to: {FILTERED_DATA_PATH}")
    
    return df_filtered

if name == "main":
    import os
    run_task1()
