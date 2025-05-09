import re
import numpy as np
import nltk
# Uncomment the following line if you need to download NLTK data
nltk.download('punkt')

# Updated import to resolve PorterStemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# A basic Vietnamese stopword list (extend as needed)
VIETNAMESE_STOPWORDS = {
    'của', 'là', 'và', 'có', 'cho', 'mình', 'cái', 'nhé', 'được',
    'đi', 'không', 'ở', 'tôi', 'anh', 'chị', 'em', 'ông', 'bà'
}

PUNCTUATION_REGEX = re.compile(r"[\d\s!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+")

# Optional synonym mappings for common variants
SYNONYMS = {
    'học phí': ['chi phí học', 'phí học'],
    'ngành': ['khoa', 'chuyên ngành'],
    # Add more domain-specific synonyms here
}


def normalize_text(text):
    """
    Lowercase, remove punctuation, and strip extra whitespace.
    """
    text = text.lower()
    # Replace punctuation and digits with spaces
    text = PUNCTUATION_REGEX.sub(' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def vietnamese_tokenizer(text):
    """
    Tokenize Vietnamese text, lowercase, remove punctuation, stopwords, and apply synonym mapping.
    """
    text = normalize_text(text)
    tokens = nltk.wordpunct_tokenize(text)
    cleaned = []
    for token in tokens:
        if token in VIETNAMESE_STOPWORDS:
            continue
        # Map synonyms to canonical form
        for canonical, variants in SYNONYMS.items():
            if token in variants:
                token = canonical
                break
        cleaned.append(token)
    return cleaned


def stem(word):
    """Stem a word to its root form."""
    return stemmer.stem(word)


def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag-of-words numpy array for the tokenized sentence.
    """
    # Stem tokens
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Example usage
if __name__ == '__main__':
    sample = "Cho mình hỏi cái này được không?"
    print("Original:", sample)
    tokens = vietnamese_tokenizer(sample)
    print("Tokens:", tokens)
    print("Stem of first token:", stem(tokens[0]))
