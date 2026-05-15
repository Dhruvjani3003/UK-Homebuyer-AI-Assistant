import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

import aiml
import pandas as pd
import numpy as np
import wikipedia
import re
import json
import tkinter as tk
import skfuzzy as fuzz
from tkinter import filedialog
from PIL import Image
from spellchecker import SpellChecker
spell = SpellChecker()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer
import nltk

from nltk.sem import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# NLTK Setup


def ensure_nltk_resources():
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_resources()
lemmatizer = WordNetLemmatizer()

#spelling correction
def correct_spelling(text):
    words = text.split()
    corrected = []
    for word in words:
        # Skip uppercase words
        if word.isupper():
            corrected.append(word)
            continue
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return " ".join(corrected)

#load aiml chatbot

kernel = aiml.Kernel()
kernel.verbose(False)
kernel.learn("homebuyer.aiml")
print("=" * 60)
print("     HOMEBUYER ASSISTANT - AI Chatbot System")
print("     Developed for ISYS30221 Artificial Intelligence")
print("=" * 60)
print("  Topics  : UK Property, Mortgages, Buying Process")
print("  Features: AIML, TF-IDF, Logic Reasoning, CNN Vision")
print("  Dataset : UK Room Classification (Bedroom, Kitchen,")
print("            Livingroom)")
print("  KB      : First Order Logic - Property Knowledge Base")
print("-" * 60)
print("  Type 'quit' to exit")
print("=" * 60)

#load Csv Q/A data

qa_data = pd.read_csv("qa_pairs.csv")
questions = qa_data["Question"].tolist()
answers = qa_data["Answer"].tolist()

# text preprocessing 

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# TF-IDF Vectorizer

vectorizer = TfidfVectorizer()
processed_questions = [preprocess(q) for q in questions]
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# similarity search

def get_similarity_response(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[0][best_match_index]
    if best_score > 0.5:
        return answers[best_match_index]
    return None

# Wikipedia fallback

def get_wikipedia_answer(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        keywords = ["property", "house", "mortgage", "buyer", 
            "loan", "stamp", "ownership", "shared", 
            "rent", "purchase", "residential"]
        if any(word in summary.lower() for word in keywords):
            return summary
        return None
    except:
        return None

# load logical knowledge base 

kb = []

with open("Logic.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            kb.append(read_expr(line))

# task b extra - load fuzzy kb 
try:
    with open("fuzzy_kb.json", "r") as f:
        fuzzy_kb = json.load(f)
    fuzzy_loaded = True
except:
    fuzzy_kb = {}
    fuzzy_loaded = False

# task b - logic reasoning 

def handle_logic(user_input):

    know_pattern = re.match(r"^i know that (.+) is (.+)$", user_input, re.I)
    check_pattern = re.match(r"^check that (.+) is (.+)$", user_input, re.I)

    if know_pattern:
        subject = know_pattern.group(1).strip().replace(" ", "_")
        category = know_pattern.group(2).strip().replace(" ", "")
        expr = read_expr(f"{category}({subject})")
        negated = read_expr(f"-({expr})")
        if ResolutionProver().prove(negated, kb):
            return "That contradicts what I already know."
        kb.append(expr)
        return f"OK, I will remember that {subject} is {category}."

    if check_pattern:
        subject = check_pattern.group(1).strip().replace(" ", "_")
        category = check_pattern.group(2).strip().replace(" ", "")
        expr = read_expr(f"{category}({subject})")
        if ResolutionProver().prove(expr, kb):
            return "Correct."
        if ResolutionProver().prove(read_expr(f"-({expr})"), kb):
            return "It may not be true... let me check... Incorrect."
        return "It may not be true... let me check... I don't know."

    return None

# task b extra - fuzzy logic 

def get_fuzzy_label(confidence):
    if confidence >= 0.85:
        return "Very Likely"
    elif confidence >= 0.65:
        return "Likely"
    elif confidence >= 0.45:
        return "Uncertain"
    elif confidence >= 0.25:
        return "Unlikely"
    else:
        return "Very Unlikely"

def handle_fuzzy_logic(user_input):
    pattern1 = re.match(
        r"^how likely is (.+) to be (.+)$",
        user_input, re.I
    )
    pattern2 = re.match(
        r"^is (.+) likely to be (.+)$",
        user_input, re.I
    )
    pattern3 = re.match(
        r"^fuzzy check (.+) is (.+)$",
        user_input, re.I
    )

    pattern = pattern1 or pattern2 or pattern3

    if not pattern:
        return None

    subject = pattern.group(1).strip().lower().replace(" ", "")
    category = pattern.group(2).strip().lower().replace(" ", "")

    key = f"{subject}_{category}"
    reverse_key = f"{category}_{subject}"

    if key in fuzzy_kb:
        confidence = fuzzy_kb[key]
    elif reverse_key in fuzzy_kb:
        confidence = fuzzy_kb[reverse_key]
    else:
        return f"I don't have fuzzy knowledge about '{subject}' being '{category}'."

    label = get_fuzzy_label(confidence)

    x = np.linspace(0, 1, 100)
    low = fuzz.trimf(x, [0, 0, 0.5])
    medium = fuzz.trimf(x, [0, 0.5, 1])
    high = fuzz.trimf(x, [0.5, 1, 1])

    low_val = fuzz.interp_membership(x, low, confidence)
    medium_val = fuzz.interp_membership(x, medium, confidence)
    high_val = fuzz.interp_membership(x, high, confidence)

    result  = f"\n  Fuzzy Logic Analysis:"
    result += f"\n  Subject   : {subject}"
    result += f"\n  Property  : {category}"
    result += f"\n  Confidence: {confidence:.2f} ({confidence*100:.0f}%)"
    result += f"\n  Verdict   : {label}"
    result += f"\n  Low       : {low_val:.2f}"
    result += f"\n  Medium    : {medium_val:.2f}"
    result += f"\n  High      : {high_val:.2f}"

    return result

# task-c load cnn model 

try:
    cnn_model = load_model("room_classifier.h5")
    with open("room_labels.json", "r") as f:
        class_labels = json.load(f)
    cnn_loaded = True
    
except Exception as e:
    cnn_loaded = False
    print(f"CNN model not loaded: {e}")

# task-c image classification logic

def handle_image_classification():
    if not cnn_loaded:
        return "Image classification is not available."
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a house image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return "No image selected."

        # Show selected image
        img_display = Image.open(file_path)
        img_display.show()

        # Preprocess for CNN
        img = keras_image.load_img(file_path, target_size=(128, 128))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = cnn_model.predict(img_array)
        predicted_index = str(np.argmax(predictions))
        predicted_label = class_labels.get(predicted_index, "Unknown")
        confidence = np.max(predictions) * 100

        # Build result with all probabilities
        result = f"This image contains: {predicted_label}\n"
        for i, prob in enumerate(predictions[0]):
            label = class_labels.get(str(i), "Unknown")
            result += f"  {label}: {prob * 100:.2f}%\n"

        return result

    except Exception as e:
        return f"Error classifying image: {e}"


# main chat loop 

while True:

    user_input = input("You: ").strip()

    # Spell correction
    corrected_input = correct_spelling(user_input)
    if corrected_input != user_input:
        print(f"  (Did you mean: '{corrected_input}'?)")
    user_input = corrected_input

    if user_input.lower() == "quit":
        print("Bot: Goodbye! Happy house hunting.")
        break

# Input validation 
    
    # Empty input
    if not user_input.strip():
        print("Bot: Please type a question or command.")
        continue

    # Numbers only
    if user_input.strip().isdigit():
        print("Bot: I can only answer homebuyer questions. Please ask me something about property or mortgages.")
        continue

    # Special characters only
    if re.match(r'^[^a-zA-Z0-9]+$', user_input.strip()):
        print("Bot: I didn't understand that. Please ask me a homebuyer related question.")
        continue

    # Very short input (1-2 chars)
    if len(user_input.strip()) <= 2:
        print("Bot: Could you please be more specific? I'm here to help with homebuyer questions.")
        continue


    # aiml pattern 
   
    aiml_response = kernel.respond(user_input.upper())
    if aiml_response == "IMAGE_CLASSIFY":
        image_response = handle_image_classification()
        print("Bot:", image_response)
        continue
    elif aiml_response:
        print("Bot:", aiml_response)
        continue

    # task b logic reasoning 

    logic_response = handle_logic(user_input)
    if logic_response:
        print("Bot:", logic_response)
        continue

    # task b extra - fuzzy logic
    
    fuzzy_response = handle_fuzzy_logic(user_input)
    if fuzzy_response:
        print("Bot:", fuzzy_response)
        continue

    # TF-IDF similarity 

    similarity_response = get_similarity_response(user_input)
    if similarity_response:
        print("Bot:", similarity_response)
        continue

    # task c - image classification 
    
    if "what is in this image" in user_input.lower() or \
       "classify image" in user_input.lower() or \
       "what type of house" in user_input.lower():
        image_response = handle_image_classification()
        print("Bot:", image_response)
        continue

    # Wikipedia fallback

    wiki_response = get_wikipedia_answer(user_input)
    if wiki_response:
        print("Bot (Wikipedia):", wiki_response)
    else:
        print("Bot: I'm not sure about that. Could you ask in another way?")