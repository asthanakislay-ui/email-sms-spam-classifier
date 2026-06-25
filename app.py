import streamlit as st
import pickle
import nltk
import string
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-font {
    font-size:42px !important;
    font-weight:bold;
    text-align:center;
}

.small-font {
    text-align:center;
    color:gray;
    font-size:18px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

.result-box{
    padding:20px;
    border-radius:12px;
    font-size:25px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------

ps = PorterStemmer()

tfidf = pickle.load(open("vectorize.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# -------------------- PREPROCESS FUNCTION --------------------


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


# -------------------- SIDEBAR --------------------

st.sidebar.title("📌 About")

st.sidebar.info("""
This application detects whether an SMS or Email is **Spam** or **Not Spam** using Machine Learning.

**Model**
- TF-IDF Vectorizer
- Multinomial Naive Bayes

Built using:
- Python
- Scikit-Learn
- Streamlit
""")

# -------------------- TITLE --------------------

st.markdown(
    "<p class='big-font'>📩 Email / SMS Spam Classifier</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='small-font'>Detect unwanted spam messages instantly using Machine Learning</p>",
    unsafe_allow_html=True
)

st.write("")

# -------------------- INPUT --------------------

input_sms = st.text_area(
    "✉️ Enter your message",
    height=180,
    placeholder="Example: Congratulations! You've won ₹10,00,000. Click here to claim..."
)

# -------------------- BUTTON --------------------

if st.button("🚀 Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:

        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        # confidence
        try:
            confidence = model.predict_proba(vector_input).max()*100
        except:
            confidence = None

        st.divider()

        if result == 1:

            st.error("🚨 SPAM DETECTED")

        else:

            st.success("✅ NOT SPAM")

        if confidence is not None:
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")

        with st.expander("🔍 Processed Text"):
            st.write(transformed_sms)

# -------------------- FOOTER --------------------

st.markdown("---")

st.caption("Built with ❤️ using Streamlit & Scikit-Learn")