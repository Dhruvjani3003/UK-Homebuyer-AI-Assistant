# UK Homebuyer AI Assistant

## Overview
The UK Homebuyer AI Assistant is an intelligent conversational system developed as part of the ISYS30221 Artificial Intelligence module at Nottingham Trent University. The project combines Natural Language Processing (NLP), First Order Logic (FOL), fuzzy reasoning, and deep learning into a single integrated AI pipeline designed to assist UK homebuyers with property-related queries.

## Key Features
- Natural language understanding using AIML and TF-IDF cosine similarity
- Spell correction using PySpellChecker with a 79.4% correction rate
- Wikipedia fallback for unknown or unanswered queries
- First Order Logic knowledge base with contradiction detection
- Fuzzy logic confidence scoring with 5 verdict levels
- CNN-based room image classification using MobileNetV2
- Fully integrated AI pipeline with 100% test success (82/82 tests passed)

## System Performance
The image classification model was trained on 4,092 room images across three categories: Bedroom, Kitchen, and Livingroom. Using MobileNetV2 transfer learning, the final model achieved an accuracy of 78.74%.

## Project Structure
| File | Description |
|---|---|
| `chatbot.py` | Main chatbot pipeline |
| `Chatboat_trining.py` | CNN model training script |
| `homebuyer.aiml` | AIML pattern rules |
| `qa_pairs.csv` | TF-IDF question-answer dataset |
| `Logic.txt` | First Order Logic expressions |
| `fuzzy_kb.json` | Fuzzy confidence knowledge base |
| `room_classifier.h5` | Trained CNN model |
| `room_labels.json` | Image classification labels |

## Technologies Used
**Programming & AI Frameworks:** Python, TensorFlow, Keras, scikit-learn, NLTK

**Logic & NLP:** AIML, scikit-fuzzy, PySpellChecker, Wikipedia API

**Data Processing:** pandas, NumPy
