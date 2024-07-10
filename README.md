# Recipe Recommendation System

## Overview

This project is a Recipe Recommendation System that suggests Indian recipes based on user queries. It uses natural language processing (NLP) and a vector database for efficient similarity searches. The system includes a preprocessing script, a backend server, and a frontend application for user interaction.

## Features

- **Preprocess data**: Cleans and preprocesses recipe data, generates embeddings, and builds an Annoy index.
- **Interactive recommendations**: Provides recipe recommendations based on user queries.
- **User-friendly interface**: Streamlit-based frontend for easy user interaction.

## Project Structure

```plaintext
.
├── preprocess_data.py
├── backend.py
├── frontend.py
├── requirements.txt
└── README.md
preprocess_data.py: Script to preprocess the dataset, generate embeddings, and build the Annoy index.
backend.py: Flask backend server that handles recipe recommendation requests.
frontend.py: Streamlit frontend application for user interaction.
requirements.txt: List of required Python packages.
README.md: Project documentation.
Setup Instructions
Prerequisites
Python 3.9.7 (managed via pyenv)
Virtual environment (recommended)
Required Python packages (listed in requirements.txt)
Installation
Clone the repository:

sh
Copy code
git clone https://github.com/valecha-a/Chatbot_with_RAG.git
cd Assignmt5
Set up a virtual environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install required packages:

sh
Copy code
pip install -r requirements.txt
Download IndianHealthyRecipe.csv file for the dataset and place it in the project directory.

Preprocess Data
Run the preprocessing script:
sh
Copy code
python preprocess_data.py
Start Backend Server
Run the backend server:
sh
Copy code
python backend.py
Start Frontend Application
Run the frontend application:
sh
Copy code
streamlit run frontend.py
Usage
Open your browser and navigate to the Streamlit application (usually http://localhost:8501).
Enter a query related to Indian recipes (e.g., "Show me a spicy Indian curry recipe").
View the top recommended dishes based on your query.
Troubleshooting
Ensure the backend server is running before starting the frontend application.
Verify that the paths to the dataset and index files are correct.
Check the console logs for any error messages and resolve the issues accordingly.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Streamlit
Flask
Sentence Transformers
Annoy






