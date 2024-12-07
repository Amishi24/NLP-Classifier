import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import streamlit as st
import torch.nn.functional as F
import speech_recognition as sr

# Define the class names and their descriptions
class_info = {
    0: {"name": "Class 0 - Normal", "description": "Statement that shows no signs of suspicion or unusual activity."},
    1: {"name": "Class 1 - Suspicious", "description": "Statement that raises some concerns or shows signs of irregularities."},
    2: {"name": "Class 2 - High Risk", "description": "Statement that strongly suggests a risk of doping or unusual behavior."},
    3: {"name": "Class 3 - Investigated", "description": "Statement that has been flagged for further investigation based on the information provided."},
    4: {"name": "Class 4 - Verified", "description": "Statement that has been confirmed to be valid and trustworthy, with no issues."},
}

# Load the model with the appropriate number of classes (5 classes in this case)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the fine-tuned model weights with strict=False to handle class mismatch
checkpoint = torch.load("model/fine_tuned_model.pth", map_location=torch.device("cpu"))
checkpoint["classifier.weight"] = checkpoint["classifier.weight"][:5, :]
checkpoint["classifier.bias"] = checkpoint["classifier.bias"][:5]

# Load the model weights
model.load_state_dict(checkpoint, strict=False)

# Function to classify text
def classify_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class index and logits
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    
    # Get the probability (softmax)
    probs = F.softmax(logits, dim=-1)
    predicted_class_prob = probs[0][predicted_class_idx].item()
    
    # Return the class info, probability, and description
    class_name = class_info.get(predicted_class_idx, {"name": "Unknown", "description": "No description available."})
    
    return class_name["name"], class_name["description"], predicted_class_prob

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("Listening for your statement...")
        recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise
        audio = recognizer.listen(source)
    
    try:
        # Recognize the speech using Google's Speech-to-Text API
        statement = recognizer.recognize_google(audio)
        st.write(f"Statement received: {statement}")
        return statement
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio. Please try again.")
        return None
    except sr.RequestError:
        st.error("There was an issue with the Speech-to-Text service. Please try again later.")
        return None

# Streamlit app setup
st.title("Anti-Doping Statement Classification")
st.subheader("Enter or Speak the statement to classify:")

# Option for user to either type or speak
input_method = st.radio("Choose input method", ("Type", "Speak"))

# If user chooses "Speak"
if input_method == "Speak":
    if st.button("Start Speaking"):
        user_statement = speech_to_text()
        if user_statement:
            class_name, description, probability = classify_text(user_statement)
            st.write(f"The statement is classified into: **{class_name}**")
            st.write(f"**Explanation**: {description}")
            st.write(f"**Model Confidence**: {probability*100:.2f}%")
            if class_name == "Class 2 - High Risk":
                st.write("This statement raises significant concerns. Further investigation is required.")
            elif class_name == "Class 3 - Investigated":
                st.write("This statement has already been flagged for further investigation. Please review the details carefully.")
            elif class_name == "Class 1 - Suspicious":
                st.write("This statement has some irregularities, but it doesn't directly indicate doping. Review the context further.")
            elif class_name == "Class 4 - Verified":
                st.write("This statement has been verified and shows no signs of manipulation or doping.")
            elif class_name == "Class 0 - Normal":
                st.write("This statement is standard and contains no unusual findings.")
else:
    # If user chooses to type the statement
    user_statement = st.text_area("Statement")
    if st.button("Classify"):
        class_name, description, probability = classify_text(user_statement)
        st.write(f"The statement is classified into: **{class_name}**")
        st.write(f"**Explanation**: {description}")
        st.write(f"**Model Confidence**: {probability*100:.2f}%")
        if class_name == "Class 2 - High Risk":
            st.write("This statement raises significant concerns. Further investigation is required.")
        elif class_name == "Class 3 - Investigated":
            st.write("This statement has already been flagged for further investigation. Please review the details carefully.")
        elif class_name == "Class 1 - Suspicious":
            st.write("This statement has some irregularities, but it doesn't directly indicate doping. Review the context further.")
        elif class_name == "Class 4 - Verified":
            st.write("This statement has been verified and shows no signs of manipulation or doping.")
        elif class_name == "Class 0 - Normal":
            st.write("This statement is standard and contains no unusual findings.")