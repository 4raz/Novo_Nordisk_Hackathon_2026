# preprocess_master.py
import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from cleaner import clean_company_name, extract_root_domain, normalize_country

RAW_CSV = "novo_vendor_master.csv"
CLEANED_CSV = "novo_vendor_master_cleaned.csv"
EMBEDDINGS_FILE = "vendor_embeddings.pt"
MODEL_NAME = "all-MiniLM-L6-v2"
NROWS = 100_000  # Adjust as needed

print(f"1. Loading raw master database ({RAW_CSV})...")
df = pd.read_csv(RAW_CSV, nrows=NROWS, dtype=str, keep_default_na=False)

print("2. Normalizing names, root domains, and ISO country codes...")
# Apply cleaner.py functions across the dataset
df["clean_name"] = df["name"].apply(clean_company_name)
df["clean_domain"] = df["website"].apply(extract_root_domain)
country_col = "country_code" if "country_code" in df.columns else "country"
df["clean_country"] = df[country_col].apply(normalize_country)

print(f"3. Saving pre-cleaned database to {CLEANED_CSV}...")
df.to_csv(CLEANED_CSV, index=False)

print(f"4. Generating vector embeddings using {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
name_list = df["clean_name"].tolist()
# Encode in batches to prevent memory overflow
embeddings = model.encode(
    name_list, batch_size=256, show_progress_bar=True, convert_to_tensor=True
)

print(f"5. Persisting tensor embeddings to {EMBEDDINGS_FILE}...")
torch.save(embeddings, EMBEDDINGS_FILE)
print("Done. Master database is fully precomputed.")
