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
                # Display Retrieval Metrics at the top
                st.subheader("Retrieval Metrics")
                st.write(f"**Precision:** {recommendations['retrieval_metrics']['precision']:.2f}")
                st.write(f"**Recall:** {recommendations['retrieval_metrics']['recall']:.2f}")
                st.write(f"**Accuracy:** {recommendations['retrieval_metrics']['accuracy']:.2f}")

                # Display Generation Metrics at the top
                st.subheader("Generation Metrics")
                st.write(f"**Faithfulness:** {recommendations['generation_metrics']['faithfulness']:.2f}")
                st.write(f"**Relevance:** {recommendations['generation_metrics']['relevance']:.2f}")
                st.write(f"**Information Integration:** {recommendations['generation_metrics']['integration']:.2f}")
                st.write(f"**Counterfactual Robustness:** {recommendations['generation_metrics']['counterfactual_robustness']:.2f}")
                st.write(f"**Negative Rejection:** {recommendations['generation_metrics']['negative_rejection']:.2f}")

                st.subheader("Recommended Dishes")
                for rec in recommendations['recommended_dishes']:
                    st.write(f"**{rec['rank']}. {rec['dish_name']} (Distance: {rec['distance']:.2f})**")
                    st.write(f"Generated Description: {rec['generated_description']}")
                    st.write("----")
            else:
                st.write(recommendations['message'])
        else:
            st.write('Error fetching recommendations.')

if __name__ == '__main__':
    main()
