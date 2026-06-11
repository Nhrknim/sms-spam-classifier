import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))



def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    y = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            y.append(ps.stem(word))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

icon = Image.open("icon.png")
st.set_page_config(
    page_title="Spam Detector",
    page_icon=icon,
    layout="centered"
)

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #ffe6f2,
            #fff5fa
        );
    }

    .title {
        color: #d63384;
        text-align: center;
        font-size: 45px;
    }

    .subtitle {
        text-align: center;
        color: #b03070;
        font-size: 18px;
    }


    textarea {
        background-color: #fff7fb !important;
        color: #4a0030 !important;
        border-radius: 15px !important;
        border: 2px solid #ff9acb !important;
    }


    .stButton button {
        background: linear-gradient(
            90deg,
            #ff69b4,
            #ff85c1
        );

        color: white;
        border-radius: 25px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }


    .stButton button:hover {
        transform: scale(1.03);
        color:white;
    }


    .stSuccess {
        border-radius: 15px;
    }

    .stError {
        border-radius: 15px;
    }


    </style>
    """,
    unsafe_allow_html=True
)

st.title("💌 Spam Detector")

input_sms = st.text_area("Enter your message",height=150,placeholder="Paste your SMS or email here...")

if st.button("✨ Analyze Message"):

    if input_sms == "":
        st.warning("Please enter a message")
    else:
        #1. Preprocess
        transform_sms = transform_text(input_sms)
        #2. Vectorize
        vector_input = tfidf.transform([transform_sms])
        #3. Predict
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]
        #4. Display
        if result == 1:
            st.header("Spam Detected")
            st.write(f"Confidence: {probability[1] * 100:.2f}%")
        else:
            st.header("Message Looks Safe")
            st.write(f"Confidence: {probability[0] * 100:.2f}%")