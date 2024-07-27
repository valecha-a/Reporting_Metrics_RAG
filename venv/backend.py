from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from annoy import AnnoyIndex
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import string

app = Flask(__name__)

# Load your dataset
recipes_df = pd.read_csv('IndianHealthyRecipe.csv')

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the Annoy Index
embedding_dim = 384  # 384 is the dimension for 'all-MiniLM-L6-v2'
annoy_index = AnnoyIndex(embedding_dim, 'angular')
embeddings = model.encode(recipes_df['Dish Name'].tolist())

for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)
annoy_index.build(10)

# Initialize the text generation model
generator = pipeline('text-generation', model='gpt2')

# Food-related keywords and accepted cuisines
food_related_keywords = ['recipe', 'cook', 'dish', 'food']
accepted_cuisines = ['indian']

def get_true_relevant_dishes(user_query):
    query_keywords = user_query.lower().split()
    relevant_dishes = recipes_df[recipes_df['Dish Name'].str.lower().apply(lambda x: any(keyword in x for keyword in query_keywords))]
    return set(relevant_dishes['Dish Name'].str.lower())

def calculate_retrieval_metrics(recommended_dishes, user_query):
    true_relevant_dishes = get_true_relevant_dishes(user_query)
    recommended_dish_names = {dish['dish_name'].lower() for dish in recommended_dishes}
    y_true = [1 if dish in true_relevant_dishes else 0 for dish in recommended_dish_names]
    y_pred = [1] * len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, accuracy

def calculate_generation_metrics(relevant_dishes, user_query):
    def faithfulness_metric(description, original_text):
        return sum([1 for word in original_text.split() if word in description.split()]) / len(original_text.split())

    def relevance_metric(description, query):
        query_words = set(query.lower().translate(str.maketrans('', '', string.punctuation)).split())
        description_words = set(description.lower().translate(str.maketrans('', '', string.punctuation)).split())
        common_words = query_words.intersection(description_words)
        return len(common_words) / len(query_words)

    def information_integration_metric(description):
        important_keys = ['ingredients', 'instructions', 'spice level', 'rating', 'dietary info']
        return sum([1 for key in important_keys if key in description.lower()]) / len(important_keys)

    def counterfactual_robustness_metric(description, altered_description):
        original_words = set(description.lower().translate(str.maketrans('', '', string.punctuation)).split())
        altered_words = set(altered_description.lower().translate(str.maketrans('', '', string.punctuation)).split())
        return 1 - (len(original_words.intersection(altered_words)) / len(original_words.union(altered_words)))

    def negative_rejection_metric(description, negative_keywords=['bad', 'worst', 'awful']):
        return sum([1 for word in description.lower().split() if word not in negative_keywords]) / len(description.split())

    query_vector = model.encode([user_query])[0]
    altered_query_vector = query_vector + np.random.normal(0, 0.1, size=query_vector.shape)
    altered_description = generator("Random altered text", max_length=300, num_return_sequences=1, truncation=True)[0]['generated_text']

    descriptions = [dish['generated_description'] for dish in relevant_dishes]
    faithfulness = np.mean([faithfulness_metric(desc, user_query) for desc in descriptions])
    relevance = np.mean([relevance_metric(desc, user_query) for desc in descriptions])
    information_integration = np.mean([information_integration_metric(desc) for desc in descriptions])
    counterfactual_robustness = np.mean([counterfactual_robustness_metric(desc, altered_description) for desc in descriptions])
    negative_rejection = np.mean([negative_rejection_metric(desc) for desc in descriptions])

    return faithfulness, relevance, information_integration, counterfactual_robustness, negative_rejection

@app.route('/interactive_recommendation', methods=['POST'])
def interactive_recommendation():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not any(keyword in user_message.lower() for keyword in food_related_keywords):
            response = {
                'message': "I am a recipe recommendation system and this question is out of my scope. Please ask me about food recipes."
            }
            return jsonify(response)

        if any(cuisine in user_message.lower() for cuisine in accepted_cuisines):
            query_vector = model.encode([user_message])[0]
            top_k = 20
            result_indices, distances = annoy_index.get_nns_by_vector(query_vector, top_k, include_distances=True)

            recommended_dishes = []
            for neighbor_index, distance in zip(result_indices, distances):
                dish = recipes_df.iloc[neighbor_index]
                dish_name = dish['Dish Name']
                description = dish.get('Description', 'No description available')
                ingredients = dish.get('Ingredients', 'No ingredients listed')
                instructions = dish.get('Instructions', 'No instructions provided')
                spice = dish.get('Spice', 'No spice information available')
                rating = dish.get('Rating', 'No rating available')
                dietary_info = dish.get('Dietary Info', 'No dietary information available')

                prompt = (f"Dish Name: {dish_name}\n"
                          f"Description: {description}\n"
                          f"Ingredients: {ingredients}\n"
                          f"Instructions: {instructions}\n"
                          f"Spice Level: {spice}\n"
                          f"Rating: {rating}\n"
                          f"Dietary Info: {dietary_info}\n\n"
                          "Detailed overview of the dish including its ingredients, preparation steps, spice level, rating, and dietary information.")
                generated_description = generator(prompt, max_length=300, num_return_sequences=1, truncation=True)[0]['generated_text']
                
                recommended_dishes.append({
                    'dish_name': dish_name,
                    'distance': distance,
                    'generated_description': generated_description
                })

            relevant_dishes = [
                dish for dish in recommended_dishes 
                if any(keyword in (dish['dish_name'] + " " + dish['generated_description']).lower() for keyword in user_message.lower().split())
            ]

            relevant_dishes = sorted(relevant_dishes, key=lambda x: x['distance'])

            for rank, dish in enumerate(relevant_dishes, start=1):
                dish['rank'] = rank

            relevant_dishes = relevant_dishes[:3]

            if not relevant_dishes:
                response = {
                    'message': f"Sorry, I couldn't find any recipes matching '{user_message}'. Please try asking in a different way or about another type of recipe."
                }
            else:
                precision, recall, accuracy = calculate_retrieval_metrics(relevant_dishes, user_message)
                faithfulness, relevance, information_integration, counterfactual_robustness, negative_rejection = calculate_generation_metrics(relevant_dishes, user_message)

                print("Recommended Dishes:", relevant_dishes)
                print("Retrieval Metrics:", {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                })

                response = {
                    'message': f"Here are the top {len(relevant_dishes)} recommended dishes for you based on '{user_message}':",
                    'recommended_dishes': relevant_dishes,
                    'retrieval_metrics': {
                        'precision': precision,
                        'recall': recall,
                        'accuracy': accuracy
                    },
                    'generation_metrics': {
                        'faithfulness': faithfulness,
                        'relevance': relevance,
                        'information_integration': information_integration,
                        'counterfactual_robustness': counterfactual_robustness,
                        'negative_rejection': negative_rejection
                    }
                }
        else:
            response = {
                'message': "I recommend only Indian dishes. Please ask me about Indian recipes."
            }

    except Exception as e:
        print(f"Error occurred: {e}")
        response = {
            'message': "An error occurred while processing your request. Please try again later."
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
