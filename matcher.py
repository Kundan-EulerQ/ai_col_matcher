import pandas as pd
import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_column_profiles(df, sample_size=5):
    """
    Extracts a semantic profile for each column combining the column name
    and a sample of unique non-null values.
    """
    profiles = {}
    for col in df.columns:
        # Get unique non-null values
        valid_vals = df[col].dropna().unique()
        
        # Sample values, handling cases with fewer values than sample_size
        if len(valid_vals) > 0:
            sample = np.random.choice(
                valid_vals, 
                size=min(sample_size, len(valid_vals)), 
                replace=False
            )
            # Convert to strings and join
            sample_str = ", ".join(map(str, sample))
        else:
            sample_str = "No data available"
            
        # Create a descriptive profile string
        profile_str = f"Column Name: {col}. Sample Data: {sample_str}"
        profiles[col] = {
            "profile_str": profile_str,
            "dtype": str(df[col].dtype),
            "sample_str": sample_str
        }
        
    return profiles

def match_columns(source_csv, target_csv, threshold=0.3):
    """
    Matches columns from source_csv to target_csv using semantic similarity.
    """
    print(f"Loading data from {source_csv} and {target_csv}...")
    try:
        df_source = pd.read_csv(source_csv, nrows=1000) # Read sample for speed
        df_target = pd.read_csv(target_csv, nrows=1000)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    print("Extracting column profiles...")
    source_profiles = extract_column_profiles(df_source)
    target_profiles = extract_column_profiles(df_target)

    print("Loading Sentence Transformer model (all-mpnet-base-v2)...")
    print("Note: First run will download the model (~420MB)")
    model = SentenceTransformer('all-mpnet-base-v2')

    print("Generating embeddings...")
    source_cols = list(source_profiles.keys())
    target_cols = list(target_profiles.keys())
    
    source_texts = [source_profiles[col]["profile_str"] for col in source_cols]
    target_texts = [target_profiles[col]["profile_str"] for col in target_cols]

    source_embeddings = model.encode(source_texts, show_progress_bar=False)
    target_embeddings = model.encode(target_texts, show_progress_bar=False)

    print("Computing similarities...")
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

    matches = []
    for i, s_col in enumerate(source_cols):
        # Find best match in target
        best_match_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_match_idx]
        
        match_info = {
            "source_column": s_col,
            "source_dtype": source_profiles[s_col]["dtype"],
            "source_sample": source_profiles[s_col]["sample_str"]
        }

        if best_score >= threshold:
            t_col = target_cols[best_match_idx]
            match_info.update({
                "target_column": t_col,
                "confidence_score": float(best_score),
                "target_dtype": target_profiles[t_col]["dtype"],
                "target_sample": target_profiles[t_col]["sample_str"]
            })
        else:
            match_info.update({
                "target_column": None,
                "confidence_score": float(best_score),
                "reason": "Score below threshold"
            })
        matches.append(match_info)

    # Sort matches by confidence score for better readability
    sorted_matches = sorted(
        matches, 
        key=lambda item: item.get('confidence_score', 0),
        reverse=True
    )

    print("\n=== MATCHING RESULTS ===\n")
    print(json.dumps(sorted_matches, indent=4))
    
    # Save to file
    output_file = "schema_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(sorted_matches, f, indent=4)
    print(f"\nMapping saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Column Matcher for ONDC Data")
    parser.add_argument("source", help="Path to Source CSV (e.g. Extract_DATA file)")
    parser.add_argument("target", help="Path to Target CSV (e.g. Test_DATA file)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum similarity threshold")
    
    args = parser.parse_args()
    
    np.random.seed(42) # For reproducible sampling
    match_columns(args.source, args.target, args.threshold)
