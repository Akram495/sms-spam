import setup  # This will download necessary NLTK files


import nltk
import os

# Set the NLTK data path manually
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer (3).pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

# Streamlit page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Inject custom CSS for background color and button styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff; /* Light blue background */
        color: #000000; /* Black text */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stTextArea textarea {
        background-color: #e0f7fa; /* Slightly lighter blue for input box */
        color: #000000; /* Black text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📧 SMS Spam Classifier")
st.markdown("### Detect if an SMS is **Spam** or **Not Spam**")
st.markdown("Enter your SMS below and click **Predict** to see the result.")

# Input area
input_sms = st.text_area("📝 Enter SMS:", height=150)

# Predict button
if st.button("🔍 Predict"):
    # Check if input is empty
    if not input_sms.strip():
        st.error("❌ Please enter a valid SMS!")
    else:
        # Preprocess and predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display results
        if result == 1:
            st.success("📩 **Spam Message**")
        else:
            st.success("📨 **Not Spam Message**")

# Add footer or explanation
st.markdown("---")

