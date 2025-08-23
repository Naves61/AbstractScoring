from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
import spacy
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Initialize models
nlp = spacy.load("en_core_sci_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\w\s-]', '', text.lower())
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def calculate_simple_score(text, config):
    if pd.isna(text) or not text.strip():
        return 0

    cleaned_text = preprocess(text)
    if not cleaned_text.strip():
        return 0

    # Keyword scoring only
    keyword_score = sum(
        weight * len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern, weight in config['keyword_weights'].items()
    )

    # Calculate final score using only the keyword score weight
    final_score = config['weights']['keyword'] * keyword_score

    return np.clip(final_score, 0, 100)


def calculate_score(text, config, target_embedding):
    if pd.isna(text) or not text.strip():
        return 0

    cleaned_text = preprocess(text)
    if not cleaned_text.strip():
        return 0

    text_embedding = sentence_model.encode([cleaned_text])[0]
    semantic_score = 1 - cosine(target_embedding, text_embedding)

    # Keyword scoring
    keyword_score = sum(
        weight * len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern, weight in config['keyword_weights'].items()
    )

    cohort_score = config['cohort_bonus'] if re.search(r'\b(n=\d+|patients\s+\d+|\d+\s+cases)\b', text) else 0
    penalty = config['case_penalty'] if re.search(r'\b(case report|single case)\b', text, flags=re.IGNORECASE) else 0

    final_score = (
        config['weights']['semantic'] * semantic_score * 100 +
        config['weights']['keyword'] * keyword_score +
        config['weights']['cohort'] * cohort_score +
        penalty
    )

    return np.clip(final_score, 0, 100)

# This is the function that should be called to calc the score for a df
def TQDMScoreCalc(df, config, target_embedding, abstract_column="ABSTRACT"):
    tqdm.pandas(desc="Grading abstracts")
    return df[abstract_column].progress_apply(lambda x: calculate_score(text=x, config=config, target_embedding=target_embedding))


def score_articles(input_file_path,output_file_path, delimiter, config_path="config.json"):
    config=load_config(config_path)

    # Create target semantic embedding
    target_embedding = sentence_model.encode([config['target_description']])[0]

    df = pd.read_csv(input_file_path, delimiter=delimiter)
    df['SCORE'] = TQDMScoreCalc(df, abstract_column="ABSTRACT", config=config, target_embedding=target_embedding)
    df.to_csv(output_file_path, index=False)
    print("Processing completed successfully")

if __name__ == '__main__':
    score_articles(input_file_path ='Articles/PM/Scored_articles/UNGRADED.csv', output_file_path ='graded_articles1.csv', delimiter =';')