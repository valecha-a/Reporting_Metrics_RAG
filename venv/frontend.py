# import streamlit as st
# import requests

# def fetch_recommendations(query_vector):
#     url = 'http://127.0.0.1:5000/recommend'  # Replace with your Flask server address if different
#     response = requests.post(url, json={'query_vector': query_vector})
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return None

# def main():
#     st.title('Recipe Recommendation System')

#     # Example query vector input (you'll replace with actual input method)
#     query_vector = [0.3] * 384

#     if st.button('Get Recommendations'):
#         recommendations = fetch_recommendations(query_vector)
#         if recommendations:
#             st.write('Top 5 Recommended Dishes:')
#             for rec in recommendations:
#                 st.write(f"{rec['rank']}. {rec['dish_name']} (Distance: {rec['distance']:.2f})")
#         else:
#             st.write('Error fetching recommendations.')

# if __name__ == '__main__':
#     main()


#----------------------------working code but not proper description------------
# import streamlit as st
# import requests

# def fetch_recommendations(query_message):
#     url = 'http://127.0.0.1:5000/interactive_recommendation'
#     response = requests.post(url, json={'message': query_message})
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return None

# def main():
#     st.title('Recipe Recommendation System')

#     user_input = st.text_input('You:')

#     if user_input:
#         recommendations = fetch_recommendations(user_input)
#         if recommendations:
#             st.write(recommendations['message'])
#             for rec in recommendations['recommended_dishes']:
#                 st.write(f"{rec['rank']}. {rec['dish_name']} (Distance: {rec['distance']:.2f})")
#                 st.write(f"    Generated Description: {rec['generated_description']}")
#         else:
#             st.write('Error fetching recommendations.')

# if __name__ == '__main__':
#     main()


# testing

import streamlit as st
import requests

def fetch_recommendations(query_message):
    url = 'http://127.0.0.1:5000/interactive_recommendation'
    response = requests.post(url, json={'message': query_message})
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    st.title('Recipe Recommendation System')

    user_input = st.text_input('You:')

    if user_input:
        recommendations = fetch_recommendations(user_input)
        if recommendations:
            if 'recommended_dishes' in recommendations:
                st.write(recommendations['message'])
                for rec in recommendations['recommended_dishes']:
                    st.write(f"{rec['rank']}. {rec['dish_name']} (Distance: {rec['distance']:.2f})")
                    st.write(f"    Generated Description: {rec['generated_description']}")
            else:
                st.write(recommendations['message'])
        else:
            st.write('Error fetching recommendations.')

if __name__ == '__main__':
    main()
