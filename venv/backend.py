# from flask import Flask, jsonify, request
# from annoy import AnnoyIndex
# import pandas as pd

# app = Flask(__name__)

# # Load the Annoy index and preprocessed data
# annoy_index = AnnoyIndex(384, 'euclidean')
# annoy_index.load('recipes_index.ann')

# recipes_df = pd.read_csv('preprocessed_recipes.csv')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # Get JSON request data
#     data = request.get_json()

#     # Example query vector (replace with actual query vector)
#     query_vector = [0.3] * 384

#     # Number of nearest neighbors to retrieve
#     top_k = 5

#     # Query the Annoy index
#     results = annoy_index.get_nns_by_vector(query_vector, top_k, include_distances=True)

#     # Prepare response with recommended dishes
#     recommended_dishes = []
#     for i, (neighbor_index, distance) in enumerate(zip(results[0], results[1])):
#         dish_name = recipes_df.iloc[neighbor_index]['Dish Name']
#         recommended_dishes.append({
#             'rank': i + 1,
#             'dish_name': dish_name,
#             'distance': distance
#         })

#     return jsonify(recommended_dishes)

# if __name__ == '__main__':
#     app.run(debug=True)


#-----------------Working code but not proper description------------- 
# from flask import Flask, jsonify, request
# from annoy import AnnoyIndex
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# app = Flask(__name__)

# # Load the Annoy index and preprocessed data
# annoy_index = AnnoyIndex(384, 'euclidean')
# annoy_index.load('recipes_index.ann')

# recipes_df = pd.read_csv('preprocessed_recipes.csv')

# # Initialize the sentence transformer model for generating embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Initialize the language model for text generation (GPT-2)
# generator = pipeline("text-generation", model="gpt2")

# @app.route('/interactive_recommendation', methods=['POST'])
# def interactive_recommendation():
#     # Get JSON request data
#     data = request.get_json()
#     user_message = data['message']

#     # Example query vector (replace with actual query vector)
#     query_vector = model.encode([user_message])

#     # Number of nearest neighbors to retrieve
#     top_k = 5

#     # Query the Annoy index
#     results = annoy_index.get_nns_by_vector(query_vector[0], top_k, include_distances=True)

#     # Prepare response with recommended dishes and generated descriptions
#     recommended_dishes = []
#     for i, (neighbor_index, distance) in enumerate(zip(results[0], results[1])):
#         dish_name = recipes_df.iloc[neighbor_index]['Dish Name']
        
#         # Generate description using language model (GPT-2)
#         generated_description = generator(f"Tell me about {dish_name}. What are its ingredients and how is it prepared?")[0]['generated_text']
        
#         recommended_dishes.append({
#             'rank': i + 1,
#             'dish_name': dish_name,
#             'distance': distance,
#             'generated_description': generated_description
#         })

#     # Simulate a chat-like response
#     response = {
#         'message': f"Here are the top {top_k} recommended dishes for you based on '{user_message}':",
#         'recommended_dishes': recommended_dishes
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)



#testing

# from flask import Flask, jsonify, request
# from annoy import AnnoyIndex
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# app = Flask(__name__)

# # Load the Annoy index and preprocessed data
# annoy_index = AnnoyIndex(384, 'euclidean')
# annoy_index.load('recipes_index.ann')

# recipes_df = pd.read_csv('preprocessed_recipes.csv')

# # Initialize the sentence transformer model for generating embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Initialize the language model for text generation (GPT-2)
# generator = pipeline("text-generation", model="gpt2")

# # Keywords to check if the query is food-related
# food_related_keywords = ["recipe", "dish", "food", "ingredient", "cooking", "meal", "cuisine", "spice", "sweet", "dessert"]

# @app.route('/interactive_recommendation', methods=['POST'])
# def interactive_recommendation():
#     # Get JSON request data
#     data = request.get_json()
#     user_message = data['message']

#     # Simple keyword check to determine if the query is related to food recipes
#     if not any(keyword in user_message.lower() for keyword in food_related_keywords):
#         response = {
#             'message': "I am a recipe recommendation system and this question is out of my scope. Please ask me about food recipes."
#         }
#         return jsonify(response)

#     # Encode the user message to get the query vector
#     query_vector = model.encode([user_message])

#     # Number of nearest neighbors to retrieve
#     top_k = 20  # Increase this to get a broader set and filter later

#     # Query the Annoy index
#     result_indices, distances = annoy_index.get_nns_by_vector(query_vector[0], top_k, include_distances=True)

#     # Prepare response with recommended dishes and generated descriptions
#     recommended_dishes = []
#     for neighbor_index, distance in zip(result_indices, distances):
#         dish = recipes_df.iloc[neighbor_index]
#         dish_name = dish['Dish Name']
#         description = dish.get('Description', 'No description available')
#         ingredients = dish.get('Ingredients', 'No ingredients listed')
#         instructions = dish.get('Instructions', 'No instructions provided')
#         spice = dish.get('Spice', 'No spice information available')
#         rating = dish.get('Rating', 'No rating available')
#         dietary_info = dish.get('Dietary Info', 'No dietary information available')

#         # Combine dish information to generate a more detailed description
#         prompt = (f"Dish Name: {dish_name}\n"
#                   f"Description: {description}\n"
#                   f"Ingredients: {ingredients}\n"
#                   f"Instructions: {instructions}\n"
#                   f"Spice Level: {spice}\n"
#                   f"Rating: {rating}\n"
#                   f"Dietary Info: {dietary_info}\n\n"
#                   "Detailed overview of the dish including its ingredients, preparation steps, spice level, rating, and dietary information.")
        
#         # Generate description with increased max tokens
#         generated_description = generator(prompt, max_new_tokens=300, num_return_sequences=1, truncation=True)[0]['generated_text']
        
#         recommended_dishes.append({
#             'dish_name': dish_name,
#             'distance': distance,
#             'generated_description': generated_description
#         })

#     # Filter and rank the dishes based on their relevance to the user query
#     relevant_dishes = [
#         dish for dish in recommended_dishes 
#         if any(keyword in (dish['dish_name'] + " " + dish['generated_description']).lower() for keyword in user_message.lower().split())
#     ]

#     # Sort relevant dishes by distance
#     relevant_dishes = sorted(relevant_dishes, key=lambda x: x['distance'])

#     # Assign ranks to the dishes
#     for rank, dish in enumerate(relevant_dishes, start=1):
#         dish['rank'] = rank

#     # Limit to top 3 dishes
#     relevant_dishes = relevant_dishes[:3]

#     # If no relevant dishes found, respond accordingly
#     if not relevant_dishes:
#         response = {
#             'message': f"Sorry, I couldn't find any recipes matching '{user_message}'. Please try asking in a different way or about another type of recipe."
#         }
#     else:
#         # Simulate a chat-like response
#         response = {
#             'message': f"Here are the top {len(relevant_dishes)} recommended dishes for you based on '{user_message}':",
#             'recommended_dishes': relevant_dishes
#         }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)

# testing more

from flask import Flask, jsonify, request
from annoy import AnnoyIndex
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)

# Load the Annoy index and preprocessed data
annoy_index = AnnoyIndex(384, 'euclidean')
annoy_index.load('recipes_index.ann')

recipes_df = pd.read_csv('preprocessed_recipes.csv')

# Initialize the sentence transformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the language model for text generation (GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Keywords to check if the query is food-related
food_related_keywords = ["recipe", "dish", "food", "ingredient", "cooking", "meal", "cuisine", "spice", "sweet", "dessert"]

# Accepted cuisines that the chatbot can recommend
accepted_cuisines = ["indian"]

@app.route('/interactive_recommendation', methods=['POST'])
def interactive_recommendation():
    # Get JSON request data
    data = request.get_json()
    user_message = data['message']

    # Simple keyword check to determine if the query is related to food recipes
    if not any(keyword in user_message.lower() for keyword in food_related_keywords):
        response = {
            'message': "I am a recipe recommendation system and this question is out of my scope. Please ask me about food recipes."
        }
        return jsonify(response)

    # Check if the user is asking for cuisines other than Indian
    if any(cuisine in user_message.lower() for cuisine in accepted_cuisines):
        # Encode the user message to get the query vector
        query_vector = model.encode([user_message])

        # Number of nearest neighbors to retrieve
        top_k = 20  # Increase this to get a broader set and filter later

        # Query the Annoy index
        result_indices, distances = annoy_index.get_nns_by_vector(query_vector[0], top_k, include_distances=True)

        # Prepare response with recommended dishes and generated descriptions
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

            # Combine dish information to generate a more detailed description
            prompt = (f"Dish Name: {dish_name}\n"
                      f"Description: {description}\n"
                      f"Ingredients: {ingredients}\n"
                      f"Instructions: {instructions}\n"
                      f"Spice Level: {spice}\n"
                      f"Rating: {rating}\n"
                      f"Dietary Info: {dietary_info}\n\n"
                      "Detailed overview of the dish including its ingredients, preparation steps, spice level, rating, and dietary information.")
            
            # Generate description with increased max tokens
            generated_description = generator(prompt, max_new_tokens=300, num_return_sequences=1, truncation=True)[0]['generated_text']
            
            recommended_dishes.append({
                'dish_name': dish_name,
                'distance': distance,
                'generated_description': generated_description
            })

        # Filter and rank the dishes based on their relevance to the user query
        relevant_dishes = [
            dish for dish in recommended_dishes 
            if any(keyword in (dish['dish_name'] + " " + dish['generated_description']).lower() for keyword in user_message.lower().split())
        ]

        # Sort relevant dishes by distance
        relevant_dishes = sorted(relevant_dishes, key=lambda x: x['distance'])

        # Assign ranks to the dishes
        for rank, dish in enumerate(relevant_dishes, start=1):
            dish['rank'] = rank

        # Limit to top 3 dishes
        relevant_dishes = relevant_dishes[:3]

        # If no relevant dishes found, respond accordingly
        if not relevant_dishes:
            response = {
                'message': f"Sorry, I couldn't find any recipes matching '{user_message}'. Please try asking in a different way or about another type of recipe."
            }
        else:
            # Simulate a chat-like response
            response = {
                'message': f"Here are the top {len(relevant_dishes)} recommended dishes for you based on '{user_message}':",
                'recommended_dishes': relevant_dishes
            }
    else:
        # Respond that the chatbot can only recommend Indian dishes
        response = {
            'message': "I recommend only Indian dishes. Please ask me about Indian recipes."
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
