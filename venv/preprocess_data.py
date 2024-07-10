# import pandas as pd

# def preprocess_data():
#     # Load the dataset
#     file_path = '/Users/anmolvalecha/Cloud Backups/PromptEngg/Assignment5_check/venv/IndianHealthyRecipe.csv'
#     recipes_df = pd.read_csv(file_path)

#     # Display the first few rows of the dataset
#     print("Initial Data Preview:")
#     print(recipes_df.head())

#     # Drop duplicates
#     recipes_df.drop_duplicates(inplace=True)

#     # Handle missing values
#     recipes_df.fillna('', inplace=True)

#     # Helper function to clean and convert time columns
#     def convert_time(value):
#         value = str(value).lower().replace('prep ', '').replace(' mins', '').strip()
#         return int(value) if value.isdigit() else 0

#     # Convert columns to appropriate data types
#     recipes_df['Prep Time'] = recipes_df['Prep Time'].apply(convert_time)
#     recipes_df['Cook Time'] = recipes_df['Cook Time'].apply(convert_time)
#     recipes_df['Rating'] = recipes_df['Rating'].apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0.0)
#     recipes_df['Number of Votes'] = recipes_df['Number of Votes'].apply(lambda x: int(x) if str(x).isdigit() else 0)
#     recipes_df['Serves'] = recipes_df['Serves'].apply(lambda x: int(x) if str(x).isdigit() else 0)
#     recipes_df['Views'] = recipes_df['Views'].apply(lambda x: int(str(x).replace(',', '')) if str(x).replace(',', '').isdigit() else 0)

#     # Further preprocessing steps as required
#     print("Preprocessed Data Preview:")
#     print(recipes_df.head())

#     # Save the preprocessed data
#     recipes_df.to_csv(file_path, index=False)

# if __name__ == "__main__":
#     preprocess_data()


#trying with milvus

# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from pymilvus import connections, Collection

# def preprocess_data():
#     # Load the dataset
#     file_path = '/Users/anmolvalecha/Cloud Backups/prompengg/Assign5/venv/IndianHealthyRecipe.csv'
#     recipes_df = pd.read_csv(file_path)

#     # Display the first few rows of the dataset
#     print("Initial Data Preview:")
#     print(recipes_df.head())

#     # Drop duplicates
#     recipes_df.drop_duplicates(inplace=True)

#     # Handle missing values
#     recipes_df.fillna('', inplace=True)

#     # Convert columns to appropriate data types and handle specific transformations
#     recipes_df['Prep Time'] = recipes_df['Prep Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Cook Time'] = recipes_df['Cook Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Rating'] = pd.to_numeric(recipes_df['Rating'], errors='coerce').fillna(0.0)
#     recipes_df['Number of Votes'] = pd.to_numeric(recipes_df['Number of Votes'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Serves'] = pd.to_numeric(recipes_df['Serves'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Views'] = pd.to_numeric(recipes_df['Views'], errors='coerce').fillna(0).astype(int)

#     # Display the preprocessed data
#     print("Preprocessed Data Preview:")
#     print(recipes_df.head())

#     # Initialize the sentence transformer model for generating embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Combine relevant text fields for embedding generation
#     recipes_df['text'] = recipes_df['Dish Name'] + ' ' + recipes_df['Ingredients'] + ' ' + recipes_df['Instructions']

#     # Generate embeddings
#     embeddings = model.encode(recipes_df['text'].tolist(), show_progress_bar=True)

#     # Connect to Milvus
#     connections.connect("default", host="localhost", port=19530)

#     # Load the Milvus collection
#     collection = Collection("healthy_recipes")

#     # Prepare data for insertion
#     data = [
#         [0] * len(embeddings),  # Placeholder for auto_id
#         recipes_df['Prep Time'].tolist(),
#         recipes_df['Cook Time'].tolist(),
#         recipes_df['Rating'].tolist(),
#         recipes_df['Number of Votes'].tolist(),
#         recipes_df['Serves'].tolist(),
#         recipes_df['Views'].tolist(),
#         embeddings.tolist()
#     ]

#     # Insert data into Milvus
#     collection.insert(data)

#     # Save the preprocessed data (if needed)
#     # recipes_df.to_csv(file_path, index=False)

#     print("Data inserted into Milvus and preprocessed data saved.")

# if __name__ == "__main__":
#     preprocess_data()



# running annoy code

# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from annoy import AnnoyIndex
# import sys
# import time  # Import time module for sleep

# def preprocess_data():
#     # Load the dataset
#     file_path = '/Users/anmolvalecha/Cloud Backups/prompengg/Assignmt5/venv/IndianHealthyRecipe.csv'
#     recipes_df = pd.read_csv(file_path)

#     # Display the first few rows of the dataset
#     print("Initial Data Preview:")
#     print(recipes_df.head())

#     # Drop duplicates
#     recipes_df.drop_duplicates(inplace=True)

#     # Handle missing values
#     recipes_df.fillna('', inplace=True)

#     # Convert columns to appropriate data types and handle specific transformations
#     recipes_df['Prep Time'] = recipes_df['Prep Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Cook Time'] = recipes_df['Cook Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Rating'] = pd.to_numeric(recipes_df['Rating'], errors='coerce').fillna(0.0)
#     recipes_df['Number of Votes'] = pd.to_numeric(recipes_df['Number of Votes'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Serves'] = pd.to_numeric(recipes_df['Serves'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Views'] = pd.to_numeric(recipes_df['Views'], errors='coerce').fillna(0).astype(int)

#     # Display the preprocessed data
#     print("Preprocessed Data Preview:")
#     print(recipes_df.head())

#     # Initialize the sentence transformer model for generating embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Combine relevant text fields for embedding generation
#     recipes_df['text'] = recipes_df['Dish Name'] + ' ' + recipes_df['Ingredients'] + ' ' + recipes_df['Instructions']

#     # Generate embeddings
#     embeddings = model.encode(recipes_df['text'].tolist(), show_progress_bar=True)

#     # Initialize Annoy index
#     dimension = 384  # Adjusted dimension to match SentenceTransformer model output
#     annoy_index = AnnoyIndex(dimension, 'euclidean')

#     # Add items to the Annoy index
#     for i, embedding in enumerate(embeddings):
#         annoy_index.add_item(i, embedding)

#     # Build the Annoy index
#     annoy_index.build(10)  # 10 trees

#     # Query the Annoy index (example query)
#     query_vector = [0.3] * dimension  # Replace with your actual query vector
#     top_k = 5  # Number of nearest neighbors to retrieve
#     results = annoy_index.get_nns_by_vector(query_vector, top_k, include_distances=True)

#     print("Query results:")
#     for i, (neighbor_index, distance) in enumerate(zip(results[0], results[1])):
#         print(f"Neighbor {i + 1}: Index {neighbor_index}, Distance {distance}")

#     print("Data indexed and queried using Annoy.")

# if __name__ == "__main__":
#     preprocess_data()



# #working
# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from annoy import AnnoyIndex
# import sys
# import time  # Import time module for sleep

# def preprocess_data():
#     # Load the dataset
#     file_path = '/Users/anmolvalecha/Cloud Backups/prompengg/Assignmt5/venv/IndianHealthyRecipe.csv'
#     recipes_df = pd.read_csv(file_path)

#     # Display the first few rows of the dataset
#     print("Initial Data Preview:")
#     print(recipes_df.head())

#     # Drop duplicates
#     recipes_df.drop_duplicates(inplace=True)

#     # Handle missing values
#     recipes_df.fillna('', inplace=True)

#     # Convert columns to appropriate data types and handle specific transformations
#     recipes_df['Prep Time'] = recipes_df['Prep Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Cook Time'] = recipes_df['Cook Time'].apply(lambda x: int(x.replace(' mins', '').strip()) if isinstance(x, str) else 0)
#     recipes_df['Rating'] = pd.to_numeric(recipes_df['Rating'], errors='coerce').fillna(0.0)
#     recipes_df['Number of Votes'] = pd.to_numeric(recipes_df['Number of Votes'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Serves'] = pd.to_numeric(recipes_df['Serves'], errors='coerce').fillna(0).astype(int)
#     recipes_df['Views'] = pd.to_numeric(recipes_df['Views'], errors='coerce').fillna(0).astype(int)

#     # Display the preprocessed data
#     print("Preprocessed Data Preview:")
#     print(recipes_df.head())

#     # Initialize the sentence transformer model for generating embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Combine relevant text fields for embedding generation
#     recipes_df['text'] = recipes_df['Dish Name'] + ' ' + recipes_df['Ingredients'] + ' ' + recipes_df['Instructions']

#     # Generate embeddings
#     embeddings = model.encode(recipes_df['text'].tolist(), show_progress_bar=True)

#     # Initialize Annoy index
#     dimension = 384  # Adjusted dimension to match SentenceTransformer model output
#     annoy_index = AnnoyIndex(dimension, 'euclidean')

#     # Add items to the Annoy index
#     for i, embedding in enumerate(embeddings):
#         annoy_index.add_item(i, embedding)

#     # Build the Annoy index
#     annoy_index.build(10)  # 10 trees

#     # Save the Annoy index to a file
#     annoy_index.save('recipes_index.ann')

#     # Optionally, you can also save the preprocessed dataframe to use later
#     recipes_df.to_csv('preprocessed_recipes.csv', index=False)

#     print("Annoy index saved to 'recipes_index.ann'.")
#     print("Preprocessed data saved to 'preprocessed_recipes.csv'.")

# if __name__ == "__main__":
#     preprocess_data()



import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import sys
import time  # Import time module for sleep

def preprocess_data():
    # Load the dataset
    file_path = '/Users/anmolvalecha/Cloud Backups/prompengg/Assignmt5/venv/IndianHealthyRecipe.csv'
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
