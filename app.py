import streamlit as st
import time
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import os

# Configure NLTK to use local data
nltk_data_path = './nltk_data'  # Local nltk data directory
nltk.data.path = [nltk_data_path] + nltk.data.path

# Streamlit page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Initialize PorterStemmer
ps = PorterStemmer()

# Ensure necessary NLTK data is available
if not os.path.exists(os.path.join(nltk_data_path, "tokenizers", "punkt")):
    nltk.download('punkt', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_path)

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english')]
    return " ".join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer files not found! Make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory as this script.")

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json")

# Splash Screen
def show_splash_screen():
    splash = st.empty()
    splash.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; background-color:#f0f8ff;">
            <h1 style="color:#4CAF50; font-size: 50px;">📧 SMS Spam Classifier</h1>
            <p style="color:#000000; font-size: 20px;">Loading the app... Please wait!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(3)
    splash.empty()

show_splash_screen()

# Option Menu for Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Feedback"],
    icons=["house", "info", "chat-dots"],
    menu_icon="menu-up",
    default_index=0,
    orientation="horizontal",
)

# Home Tab
if selected == "Home":
    st.title("📧 SMS Spam Classifier")
    st.markdown("### Detect if an SMS is **Spam** or **Not Spam**")
    st.markdown("Enter your SMS below and click **Predict** to see the result.")

    # Input area
    input_sms = st.text_area("📝 Enter SMS:", height=150)

    # Predict button
    if st.button("🔍 Predict"):
        if not input_sms.strip():
            st.error("❌ Please enter a valid SMS!")
        else:
            try:
                # Preprocess and predict
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                # Display results
                if result == 1:
                    st.success("📩 **Spam Message**")
                else:
                    st.success("📨 **Not Spam Message**")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

# About Tab
elif selected == "About":
    st.header("About the App")
    st.markdown("""
    This app uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to classify SMS messages as either **Spam** or **Not Spam**.

    ### Features:
    - Preprocessing of SMS messages
    - Spam detection using a trained ML model
    - User-friendly interface

    ### How It Works:
    1. The user enters an SMS message.
    2. The app preprocesses the text (e.g., removes stopwords, stems words).
    3. The message is classified using a trained machine learning model.
    """)

# Feedback Tab
elif selected == "Feedback":
    st.header("Feedback")
    st.markdown("We value your feedback! Let us know your thoughts below:")
    feedback = st.text_area("Write your feedback here:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
