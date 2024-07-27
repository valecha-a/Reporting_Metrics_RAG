# Recipe Recommendation System with RAG

## Youtube Video link: https://youtu.be/Fh0Y5mWIreE

## Project Description

The Recipe Recommendation System is a web application that leverages Retrieval-Augmented Generation (RAG) for providing personalized recipe recommendations. It utilizes natural language processing and a vector database to match user queries with relevant recipes from the dataset. The system integrates:

- **Flask** for the backend API
- **Sentence Transformers** for generating embeddings
- **Annoy** for efficient similarity search
- **Transformers (GPT-2)** for generating detailed dish descriptions
- **Streamlit** for a user-friendly frontend interface

## Features

- **Recipe Recommendation:** Provides recommendations based on user queries.
- **Retrieval Metrics:** Calculates precision, recall, and accuracy for recommendations.
- **Generation Metrics:** Evaluates the quality of generated descriptions based on several criteria.

## Installation

To set up the Recipe Recommendation System on your local machine, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/valecha-a/Reporting_Metrics_RAG.git
   cd Reporting_Metrics_RAG
Create a Virtual Environment:

bash
Copy code
python -m venv venv
Activate the Virtual Environment:

On macOS/Linux:

bash
Copy code
source venv/bin/activate
On Windows:

bash
Copy code
venv\Scripts\activate
Install Required Packages:

bash
Copy code
pip install -r requirements.txt
Prepare the Dataset:

Ensure that the IndianHealthyRecipe.csv file is located in the root directory of the project.

Usage
Run the Flask Backend:

bash
Copy code
python app.py
The Flask API will be available at http://localhost:5000.

Run the Streamlit Frontend:

bash
Copy code
streamlit run frontend.py
The Streamlit application will be accessible at http://localhost:8501.

## Interacting with the System:

Open the Streamlit application in your web browser.
Enter your recipe query in the text input field and click "Get Recommendations."
The application will display recommended dishes along with retrieval and generation metrics.
Code Structure
app.py: The main Flask application handling the backend logic, including recipe recommendation and metric calculations.
frontend.py: The Streamlit application serving as the frontend for user interaction.
IndianHealthyRecipe.csv: The dataset used for recipe recommendations.
Metrics
Retrieval Metrics
Precision: The fraction of relevant dishes among the recommended dishes.
Recall: The fraction of relevant dishes that are retrieved.
Accuracy: The proportion of correctly recommended dishes.
Generation Metrics
Faithfulness: Measures how closely the generated description matches the original query.
Relevance: Evaluates how well the generated description aligns with the query.
Information Integration: Assesses how well the description incorporates key recipe information.
Counterfactual Robustness: Measures the resilience of the description to alterations.
Negative Rejection: Evaluates the presence of negative keywords in the description.

sh
Copy code
git clone https://github.com/valecha-a/Chatbot_with_RAG.git
cd Assignmt6
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

## Usage
Open your browser and navigate to the Streamlit application (usually http://localhost:8501).
Enter a query related to Indian recipes (e.g., "Show me a spicy Indian curry recipe").
View the top recommended dishes based on your query.

## Troubleshooting
Ensure the backend server is running before starting the frontend application.
Verify that the paths to the dataset and index files are correct.
Check the console logs for any error messages and resolve the issues accordingly.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Streamlit
Flask
Sentence Transformers
Annoy






