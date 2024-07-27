import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import sys
import time  # Import time module for sleep

def preprocess_data():
    # Load the dataset
    file_path = '/Users/anmolvalecha/Cloud Backups/prompengg/Assignment6/venv/IndianHealthyRecipe.csv'
    recipes_df = pd.read_csv(file_path)

    # Display the first few rows of the dataset
    print("Initial Data Preview:")
    print(recipes_df.head())

    # Drop duplicates
    recipes_df.drop_duplicates(inplace=True)

    # Handle missing values
    recipes_df.fillna('', inplace=True)

    # Convert columns to appropriate data types and handle specific transformations
    recipes_df['Prep Time'] = recipes_df['Prep Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
    recipes_df['Cook Time'] = recipes_df['Cook Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
    recipes_df['Rating'] = pd.to_numeric(recipes_df['Rating'], errors='coerce').fillna(0.0)
    recipes_df['Number of Votes'] = pd.to_numeric(recipes_df['Number of Votes'], errors='coerce').fillna(0).astype(int)
    recipes_df['Serves'] = pd.to_numeric(recipes_df['Serves'], errors='coerce').fillna(0).astype(int)
    recipes_df['Views'] = pd.to_numeric(recipes_df['Views'], errors='coerce').fillna(0).astype(int)

    # Display the preprocessed data
    print("Preprocessed Data Preview:")
    print(recipes_df.head())

    # Initialize the sentence transformer model for generating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combine relevant text fields for embedding generation
    recipes_df['text'] = recipes_df['Dish Name'] + ' ' + recipes_df['Ingredients'] + ' ' + recipes_df['Instructions']

    # Generate embeddings
    embeddings = model.encode(recipes_df['text'].tolist(), show_progress_bar=True)

    # Initialize Annoy index
    dimension = 384  # Adjusted dimension to match SentenceTransformer model output
    annoy_index = AnnoyIndex(dimension, 'euclidean')

    # Add items to the Annoy index
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    # Build the Annoy index
    annoy_index.build(10)  # 10 trees

    # Save the Annoy index to a file
    annoy_index.save('recipes_index.ann')

    # Optionally, you can also save the preprocessed dataframe to use later
    recipes_df.to_csv('preprocessed_recipes.csv', index=False)

    print("Annoy index saved to 'recipes_index.ann'.")
    print("Preprocessed data saved to 'preprocessed_recipes.csv'.")

if __name__ == "__main__":
    preprocess_data()
