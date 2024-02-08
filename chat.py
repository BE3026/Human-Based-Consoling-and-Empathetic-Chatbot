from openai import OpenAI
import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

st.title("User Consoling Via Text")
def detect_sentiment(message):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(message)['compound']
    if sentiment_score >= 0.05:
        return "positive"
    elif sentiment_score <= -0.05:
        return "negative"
    else:
        return "neutral"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"]=="user":
            st.markdown(message["Display"])
        else:
            st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    print(prompt)
    sentiment=detect_sentiment(prompt)
    userinp=prompt
    prompt=prompt+" (Emotion Detected :- {})".format(sentiment)
    if sentiment=='negative':
        userinp='I am saying the next sentence in a negative emotion , so try to motivate me on the situation and ask whether you could tell me a joke if i say yes tell a joke or else skip.\n' +userinp 
    st.session_state.messages.append({"role": "user", "content": userinp,"Display":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response =st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})