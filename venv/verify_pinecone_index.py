import os
from pinecone import Pinecone

def verify_index(api_key, index_name):
    # Initialize Pinecone with your API key
    pc = Pinecone(api_key=api_key)

    try:
        print("Connected to Pinecone.")
        # Check if the index exists
        if index_name in pc.list_indexes().names():
            print(f"Index '{index_name}' exists.")
        else:
            print(f"Index '{index_name}' does not exist.")

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        print("Disconnected from Pinecone.")

if __name__ == "__main__":
    # Fetch Pinecone API key from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
    else:
        index_name = 'healthy-recipes'  # Replace with your actual index name
        verify_index(api_key, index_name)
