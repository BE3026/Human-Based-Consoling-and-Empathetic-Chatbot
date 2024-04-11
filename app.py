import streamlit as st
import webbrowser
def main():
    st.set_page_config(
    page_title="Consoling Chatbot"
)
    # Set page title
    st.title("Human Emotion Based Consoling And Empathetic Chatbot")

    # Description of the demo
    st.write("""
    Welcome to the Emotion Based Consoling Chatbot Demo! This demo showcases a chatbot that can detect emotions from both facial expressions and text messages. The chatbot utilizes machine learning models to analyze facial features and text input to determine the user's emotions.

    ### How it works:
    - **Facial Recognition:** The chatbot uses computer vision techniques to analyze facial expressions in real-time. It identifies key facial features such as eyes, mouth, and eyebrows to recognize emotions like happiness, sadness, anger, etc.
    
    - **Text Analysis:** In addition to facial recognition, the chatbot also analyzes text messages entered by the user. It uses natural language processing (NLP) techniques to understand the sentiment conveyed in the text, such as positive, negative, or neutral.

    ### Features:
    - **Real-time Emotion Detection:** Users can interact with the chatbot through text messages or webcam input, and the chatbot will instantly recognize their emotions.
    
    - **Interactive Interface:** The demo provides an interactive interface where users can type messages or enable their webcam to see real-time emotion detection.

    ### Technologies Used:
    - Python
    - OpenCV (for facial capturing)
    - CNN (for face identification)
    - NLTK (for semantic analysis)
    - Haar Cascade Classifier (for detecting emotions)
    - Streamlit (for building the web interface)
    """)

    st.write("\n")
    if st.button("Go By Video"):
        webbrowser.open("http://localhost:8501/Go_By_Video",new=0)
    if st.button("Go By Text"):
        webbrowser.open("http://localhost:8501/Go_By_Text",new=0)
if __name__ == "__main__":
    main()
