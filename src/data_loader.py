"""
Stage 1: Data Exploration, Cleaning, and Preprocessing.

This module handles loading the Amazon Fine Food Reviews dataset (or generating a high-quality sample if not provided),
cleaning duplicate entries, stripping HTML tags, filtering short/empty reviews, and creating a balanced/sampled dataset.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_sample_amazon_reviews(num_samples: int = 20000) -> pd.DataFrame:
    """
    Generates a realistic Amazon Fine Food Reviews dataset sample with diverse brands,
    categories, packaging complaints, quality feedback, and ratings.
    Matches standard Kaggle Amazon Fine Food Reviews schema.
    """
    np.random.seed(42)
    
    brands = {
        "Coffee": ["DarkRoast Co", "BeanSupreme", "MorningBrew", "JavaGold", "K-Cup Express", "Peet's Select", "Starbucks Reserve", "Nespresso Compatible"],
        "Tea": ["GreenLeaf Teas", "ChaiMaster", "HerbalHaven", "ZenTea", "EarlGrey Select", "Twinings Imperial", "Matcha Zen"],
        "Snacks": ["CrispyChips", "OrganicNuts", "FruitCrunch", "SnackBite", "VeggieCrisp", "KindBar Plus", "TrailMix Co"],
        "Pet Food": ["BarkDelight", "PurrfectBites", "HealthyPaws", "KibbleKing", "BlueBuffalo Choice", "TasteOfWild"],
        "Chocolate": ["CocoaLux", "DarkBite", "ChocoSweet", "ArtisanCacao", "Lindt Reserve", "Ghirardelli Supreme"]
    }
    
    pos_adjectives = ["amazing", "fresh", "outstanding", "delicious", "top-tier", "crisp", "rich", "delightful", "incomparable", "smooth"]
    neg_adjectives = ["stale", "rancid", "terrible", "disappointing", "awful", "bland", "overpriced", "spoiled", "bitter", "crushed"]
    
    pos_details = [
        "The aroma as soon as you open the seal is incredible.",
        "Comes packaged very neatly with a long expiration date.",
        "Perfect addition to my daily breakfast routine.",
        "Shipped fast in secure bubble wrap with no damage.",
        "Organic ingredient quality really shows in the taste profile.",
        "Great value bundle for the price, buying the 6-pack next time."
    ]
    
    neg_details = [
        "The cardboard box arrived crushed and powder leaked everywhere inside.",
        "Expiration date was only 3 days away when it arrived at my doorstep!",
        "Tastes like harsh chemical additives and leaves an aftertaste.",
        "Keurig pods fail to puncture properly, flooding my coffee maker with grounds.",
        "Way smaller quantity than shown in the product pictures. Total rip-off.",
        "Plastic seal was broken upon arrival. Unsafe to consume."
    ]
    
    records = []
    cat_keys = list(brands.keys())
    
    for i in range(num_samples):
        cat = cat_keys[i % len(cat_keys)]
        brand_list = brands[cat]
        brand = brand_list[i % len(brand_list)]
        score = int(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.10, 0.12, 0.25, 0.38]))
        
        pos_adj = pos_adjectives[i % len(pos_adjectives)]
        neg_adj = neg_adjectives[i % len(neg_adjectives)]
        pos_det = pos_details[i % len(pos_details)]
        neg_det = neg_details[i % len(neg_details)]
        
        item_id = (i % 80) + 1
        
        if score in [1, 2]:
            summary = f"{neg_adj.capitalize()} {cat.lower()} experience - review #{i+1}"
            text = f"[Review #{i+1}] I purchased the {brand} {cat} expecting quality. However, it was {neg_adj}! {neg_det} I contacted customer service but got no resolution. Rating: {score}/5."
        elif score == 3:
            summary = f"Average {cat.lower()} from {brand}"
            text = f"[Review #{i+1}] The {brand} {cat} is average. It is neither terrible nor {pos_adj}. {pos_det} But {neg_det.lower()} Given the price, 3 stars."
        else:
            summary = f"{pos_adj.capitalize()} {cat.lower()} by {brand}!"
            text = f"[Review #{i+1}] This {brand} {cat} is {pos_adj}! I've tried many items in category {cat}, but this one stands out. {pos_det} Rating: {score}/5."
            
        # Add slight HTML artifact to simulate real raw Amazon web scraped reviews (<br />)
        if i % 9 == 0:
            text = text + " <br /><br /> Thanks Amazon!"
            
        # Create intentional duplicate reviews (approx 1% rate)
        if i > 200 and i % 90 == 0:
            text = records[i - 50]["Text"]
            summary = records[i - 50]["Summary"]
            
        num_help = int(np.random.exponential(scale=2))
        den_help = num_help + int(np.random.poisson(lam=1))
        
        records.append({
            "Id": i + 1,
            "ProductId": f"B00{hash(brand + cat) % 100000:05d}",
            "UserId": f"A{i % 4000:06d}",
            "ProfileName": f"Reviewer_{i % 4000}",
            "HelpfulnessNumerator": num_help,
            "HelpfulnessDenominator": den_help,
            "Score": score,
            "Time": 1354000000 + i * 1800,
            "Summary": summary,
            "Text": text,
            "Category": cat,
            "Brand": brand
        })
        
    return pd.DataFrame(records)


def clean_html_tags(text: str) -> str:
    """Strips HTML tags like <br />, <p>, <div> from review text."""
    if not isinstance(text, str):
        return ""
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def load_and_preprocess_reviews(
    csv_path: Optional[str] = None,
    sample_size: int = 20000,
    min_char_len: int = 15,
    min_word_count: int = 3
) -> Tuple[pd.DataFrame, dict]:
    """
    Stage 1 Core Function:
    Loads, cleans, deduplicates, and filters Amazon Fine Food Reviews dataset.

    Returns:
        processed_df: Cleaned dataframe ready for embedding and RAG indexing.
        stats: Dictionary containing cleaning pipeline metrics.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[Stage 1] Loading custom dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print(f"[Stage 1] CSV path not found or not provided. Generating synthetic 25,000 Amazon review dataset...")
        df = generate_sample_amazon_reviews(num_samples=25000)
        
    initial_count = len(df)
    
    # Ensure standard column existence
    required_cols = ["Text", "Score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataframe.")
            
    if "Summary" not in df.columns:
        df["Summary"] = ""
    df["Summary"] = df["Summary"].fillna("")
    
    # 1. Clean HTML tags from Text and Summary
    df["CleanText"] = df["Text"].astype(str).apply(clean_html_tags)
    df["CleanSummary"] = df["Summary"].astype(str).apply(clean_html_tags)
    
    # Combine Summary and Text for richer embedding context
    df["CombinedReview"] = df["CleanSummary"] + " - " + df["CleanText"]
    df["CombinedReview"] = df["CombinedReview"].str.strip(" - ")
    
    # 2. Filter null or empty reviews
    df = df[df["CleanText"].str.strip() != ""].copy()
    null_filtered_count = initial_count - len(df)
    
    # 3. Deduplicate exact duplicate reviews (same CleanText)
    dup_mask = df.duplicated(subset=["CleanText"], keep="first")
    duplicates_removed = dup_mask.sum()
    df = df[~dup_mask].copy()
    
    # 4. Filter short / low-information reviews
    df["CharLen"] = df["CleanText"].str.len()
    df["WordCount"] = df["CleanText"].str.split().str.len()
    
    short_mask = (df["CharLen"] < min_char_len) | (df["WordCount"] < min_word_count)
    short_removed = short_mask.sum()
    df = df[~short_mask].copy()
    
    # 5. Compute Helpfulness Ratio & Sentiment Category
    if "HelpfulnessNumerator" in df.columns and "HelpfulnessDenominator" in df.columns:
        df["HelpfulnessDenominator"] = df["HelpfulnessDenominator"].replace(0, 1)
        df["HelpfulnessRatio"] = (df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]).round(2)
    else:
        df["HelpfulnessRatio"] = 0.0
        
    def assign_sentiment(score: int) -> str:
        if score <= 2:
            return "Negative"
        elif score == 3:
            return "Neutral"
        else:
            return "Positive"
            
    df["Sentiment"] = df["Score"].apply(assign_sentiment)
    
    # 6. Sampling if dataset is larger than requested sample_size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        
    df["ReviewId"] = df.index
    
    stats = {
        "initial_raw_count": initial_count,
        "null_or_empty_removed": null_filtered_count,
        "duplicates_removed": duplicates_removed,
        "short_reviews_removed": short_removed,
        "final_processed_count": len(df),
        "sentiment_distribution": df["Sentiment"].value_counts().to_dict(),
        "score_distribution": df["Score"].value_counts().to_dict()
    }
    
    print(f"[Stage 1 Complete] Raw: {initial_count} | Duplicates: {duplicates_removed} | Short: {short_removed} | Final: {len(df)}")
    return df, stats


if __name__ == "__main__":
    df, stats = load_and_preprocess_reviews(sample_size=1000)
    print("Stats:", stats)
    print("Sample row:", df.iloc[0][["ReviewId", "Score", "Sentiment", "CombinedReview"]])
