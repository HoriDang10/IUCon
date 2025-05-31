import random
import json
import torch
import re
from model import NeuralNet
from nlp import vietnamese_tokenizer, bag_of_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and questions
with open('Intents.json', 'r') as f:
    intents = json.load(f)
with open('Questions.json', 'r') as f:
    questions_data = json.load(f)

# Build canonical question-answer mapping
canonical_questions = []
qa_mapping = []  # list of (question, answer)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        canonical_questions.append(pattern)
        qa_mapping.append((pattern, random.choice(intent['responses'])))
for qset in questions_data['questions']:
    for qa in qset['questions_and_answers']:
        qs = qa['question'] if isinstance(qa['question'], list) else [qa['question']]
        for q in qs:
            canonical_questions.append(q)
            qa_mapping.append((q, qa['answer']))

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode', lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(canonical_questions)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, output_size, hidden_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "IU Consultant"

def retrieve_answer(sentence):
    # Normalize user input
    sent_norm = " ".join(vietnamese_tokenizer(sentence))
    # TF-IDF similarity
    vec = tfidf_vectorizer.transform([sent_norm])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idx = sims.argmax()
    if sims[idx] >= 0.7:
        return qa_mapping[idx][1]
    # Fuzzy match (full and partial)
    fuzz_scores = []
    sent_lower = sent_norm.lower()
    for i, q in enumerate(canonical_questions):
        score_full = fuzz.ratio(sent_lower, q.lower())
        score_partial = fuzz.partial_ratio(sent_lower, q.lower())
        fuzz_scores.append((i, max(score_full, score_partial)))
    best_i, best_score = max(fuzz_scores, key=lambda x: x[1])
    if best_score >= 60:
        return qa_mapping[best_i][1]
    return None


def process_chatbot_response(sentence):
    
    # 0) Year-based score query: “ngành X năm YYYY lấy bao nhiêu điểm”
    m_year = re.search(
        r"ngành\s+(.+?)\s+năm\s+(\d{4}).*lấy bao nhiêu điểm",
        sentence,
        re.IGNORECASE
    )
    if m_year:
        major = m_year.group(1).strip()
        year = m_year.group(2).strip()
        with open('Scores.json','r') as f:
            score_data = json.load(f)['major']
        if major in score_data:
            parts = []
            for method_key, val in score_data[major].items():
                parts.append(f"phương thức {method_key[-1]}: {val} điểm")
            return f"Điểm chuẩn ngành {major} năm {year}: " + "; ".join(parts)
        else:
            return f"Xin lỗi, không tìm thấy ngành {major} trong dữ liệu điểm chuẩn."

    # 1) Method-based score query: “theo phương thức X”
    m_method = re.search(
        r"điểm chuẩn.*?ngành\s+(.+?)\s+theo phương thức\s+(\d)",
        sentence,
        re.IGNORECASE
    )
    if m_method:
        major = m_method.group(1).strip()
        method = m_method.group(2).strip()
        with open('../resources/Scores.json','r') as f:
            score_data = json.load(f)['major']
        key = f"method{method}"
        if major in score_data and key in score_data[major]:
            val = score_data[major][key]
            return f"Điểm chuẩn của ngành {major} theo phương thức {method} là {val} điểm."
        else:
            return f"Xin lỗi, mình không tìm thấy điểm chuẩn của ngành {major} theo phương thức {method}."
    # 2) Retrieval (TF-IDF → Fuzzy)
    answer = retrieve_answer(sentence)
    if answer:
        return answer

    # 3) Neural-intent fallback (skip greeting)
    tokens = vietnamese_tokenizer(sentence)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X.reshape(1, -1)).to(device)
    out = model(X)
    _, pred = torch.max(out, dim=1)
    tag = tags[pred.item()]
    prob = torch.softmax(out, dim=1)[0][pred.item()].item()
    if prob > 0.75 and tag != "greeting":
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    # 4) Final fallback
    return(f"{bot_name}: Mình chưa hiểu ý của bạn...")


def get_ai_response(user_input: str) -> str:
    return process_chatbot_response(user_input)
