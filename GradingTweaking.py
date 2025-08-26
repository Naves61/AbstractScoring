import pandas as pd
import json
import re
import spacy
import random
import CSVgrader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ------------------------
# CONFIG FILE HANDLING
# ------------------------
def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

# ------------------------
# INITIALIZE NLP MODELS
# ------------------------
nlp = spacy.load("en_core_sci_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# ------------------------
# TARGET EMBEDDING & PREPROCESSING
# ------------------------
def get_target_embedding(config):
    return sentence_model.encode([config['target_description']])[0]


def preprocess(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\w\s-]', '', str(text).lower())
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])


# ------------------------
# SCORING FUNCTION
# ------------------------
def compute_scores(df, config):
    target_embedding = get_target_embedding(config)
    # df['SCORE'] = df['ABSTRACT'].apply(lambda x: calculate_score(x, config, target_embedding))
    df['SCORE'] = CSVgrader.TQDMScoreCalc(df, abstract_column="ABSTRACT", config=config, target_embedding=target_embedding)
    return df


# ------------------------
# CHECK SEPARATION
# ------------------------
def check_separation(df, score_column, included_column):
    """
    Checks if there is perfect separation between included and non-included scores.
    """
    included_scores = df[df[included_column] == 1][score_column]
    excluded_scores = df[df[included_column] == 0][score_column]

    # Check if the minimum included score is greater than the maximum excluded score
    return included_scores.min() > excluded_scores.max()


def identify_errors(df):
    errors = []
    ones = df[df['INCLUDED'] == 1]
    zeros = df[df['INCLUDED'] == 0]
    for _, row0 in zeros.iterrows():
        for _, row1 in ones.iterrows():
            if row0['SCORE'] >= row1['SCORE']:
                errors.append({
                    'zero_id': row0.name,
                    'zero_score': row0['SCORE'],
                    'zero_abstract': row0['ABSTRACT'],
                    'one_id': row1.name,
                    'one_score': row1['SCORE'],
                    'one_abstract': row1['ABSTRACT'],
                    'direction': 'zero should be lower / one should be higher'
                })
    return errors


# ------------------------
# NEW KEYWORD EXTRACTION
# ------------------------
def get_candidate_keywords(text):
    """Extract candidate keywords (noun chunks) from a text."""
    doc = nlp(str(text))
    return [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 3]


def count_keyword_frequency(keyword, abstracts):
    """Count in how many abstracts the keyword appears."""
    # Only returns the first occurrence per abstract
    # return sum(1 for abs in abstracts if keyword in abs.lower())
    return sum(abs.lower().count(keyword) for abs in abstracts)



def add_new_keywords_sparingly(error_entry, config, included_abstracts, non_included_abstracts):
    """
    Sparingly add one positive and one negative keyword.
    Positive: most frequent keyword from the eligible (included) abstract that never appears in non-included abstracts.
    Negative: most frequent keyword from the non-eligible abstract that never appears in included abstracts.
    """
    global all_included_abstracts, all_non_included_abstracts

    # For positive candidate from eligible abstract
    pos_candidates = get_candidate_keywords(error_entry['one_abstract'])
    pos_filtered = {}
    for kw in pos_candidates:
        if all(kw not in abs_text.lower() for abs_text in all_non_included_abstracts):
            freq = count_keyword_frequency(kw, all_included_abstracts)
            pos_filtered[kw] = freq
    best_pos = max(pos_filtered, key=pos_filtered.get) if pos_filtered else None

    # For negative candidate from non-eligible abstract
    neg_candidates = get_candidate_keywords(error_entry['zero_abstract'])
    neg_filtered = {}
    for kw in neg_candidates:
        if all(kw not in abs_text.lower() for abs_text in all_included_abstracts):
            freq = count_keyword_frequency(kw, all_non_included_abstracts)
            neg_filtered[kw] = freq
    best_neg = max(neg_filtered, key=neg_filtered.get) if neg_filtered else None

    # Add the keywords if not already present in config, using one per type
    good_weight = 10
    bad_weight = -10
    if best_pos:
        pattern = r'\b' + re.escape(best_pos) + r'\b'
        if pattern not in config['keyword_weights']:
            print(f"Adding GOOD keyword: '{best_pos}' with pattern {pattern} and weight {good_weight}")
            config['keyword_weights'][pattern] = good_weight
    if best_neg:
        pattern = r'\b' + re.escape(best_neg) + r'\b'
        if pattern not in config['keyword_weights']:
            print(f"Adding BAD keyword: '{best_neg}' with pattern {pattern} and weight {bad_weight}")
            config['keyword_weights'][pattern] = bad_weight

    return config

# FROM HERE DEPRECATED

def advanced_keyword_extractor(abstract_incl, abstract_nonincl):
    """
    Compares the current non-included entry to all known non-included abstracts to find
    keywords that are present in a high percentage of non-included abstracts without appearing in any of
    the included abstracts (list_bad_keywords).

    Similarly, compares the current includible entry to all known includible abstracts to obtain
    keywords that appear in a high percentage of includible abstracts and in none of the non-included abstracts (list_good_keywords).

    Uses a flexible threshold: starting at 100% and lowering it stepwise until at least one candidate is found.

    Returns:
        list_good_keywords, list_bad_keywords
    """
    global all_included_abstracts, all_non_included_abstracts

    def extract_candidates(text):
        doc = nlp(str(text))
        return set(chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 3)

    candidates_incl = extract_candidates(abstract_incl)
    candidates_nonincl = extract_candidates(abstract_nonincl)

    list_good_keywords = []
    list_bad_keywords = []

    for threshold in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for kw in candidates_incl:
            count_incl = sum(1 for text in all_included_abstracts if kw in str(text).lower())
            count_nonincl = sum(1 for text in all_non_included_abstracts if kw in str(text).lower())
            if count_incl >= threshold * len(all_included_abstracts) and count_nonincl == 0:
                if kw not in list_good_keywords:
                    list_good_keywords.append(kw)
        for kw in candidates_nonincl:
            count_nonincl = sum(1 for text in all_non_included_abstracts if kw in str(text).lower())
            count_incl = sum(1 for text in all_included_abstracts if kw in str(text).lower())
            if count_nonincl >= threshold * len(all_non_included_abstracts) and count_incl == 0:
                if kw not in list_bad_keywords:
                    list_bad_keywords.append(kw)
        if list_good_keywords or list_bad_keywords:
            break

    return list_good_keywords, list_bad_keywords


def extract_candidate_keywords(abstract_incl, abstract_nonincl):
    """
    A simple extractor that returns candidate keywords present in the includible abstract
    but not in the non-includible abstract.
    """
    doc_incl = nlp(str(abstract_incl))
    doc_nonincl = nlp(str(abstract_nonincl))

    incl_chunks = set(chunk.text.lower() for chunk in doc_incl.noun_chunks)
    nonincl_chunks = set(chunk.text.lower() for chunk in doc_nonincl.noun_chunks)

    candidates = incl_chunks - nonincl_chunks
    candidates = {kw for kw in candidates if len(kw) > 3}
    return list(candidates)


def add_new_keywords(error_entry, config):
    """
    For an error entry, first try the advanced keyword extractor.
    If advanced extraction returns no candidates, fall back to the basic extraction.
    New keywords are added to the config with an initial weight.
    """
    good_keywords, bad_keywords = advanced_keyword_extractor(error_entry['one_abstract'],
                                                             error_entry['zero_abstract'])
    if not good_keywords and not bad_keywords:
        fallback_candidates = extract_candidate_keywords(error_entry['one_abstract'],
                                                         error_entry['zero_abstract'])
        good_keywords = fallback_candidates

    good_weight = 2
    bad_weight = -2
    for kw in good_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if pattern not in config['keyword_weights']:
            print(f"Adding GOOD keyword: '{kw}' with pattern {pattern} and weight {good_weight}")
            config['keyword_weights'][pattern] = good_weight
    for kw in bad_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if pattern not in config['keyword_weights']:
            print(f"Adding BAD keyword: '{kw}' with pattern {pattern} and weight {bad_weight}")
            config['keyword_weights'][pattern] = bad_weight

    return config


# ------------------------
# MONTE CARLO PARAMETER TWEAKING FOR EACH ERROR
# ------------------------
def monte_carlo_tweak(error_entry, config, max_iterations=50, tweak_factor=0.1):
    """
    For a single error entry, perturb the parameters iteratively until the error for this pair is minimized.
    """
    best_config = json.loads(json.dumps(config))
    best_error = error_entry['zero_score'] - error_entry['one_score']

    for _ in range(max_iterations):
        candidate = json.loads(json.dumps(best_config))
        candidate['weights']['semantic'] *= (1 + random.uniform(-tweak_factor, tweak_factor))
        candidate['weights']['keyword'] *= (1 + random.uniform(-tweak_factor, tweak_factor))
        candidate['weights']['cohort'] *= (1 + random.uniform(-tweak_factor, tweak_factor))
        if 'case_penalty' in candidate:
            candidate['case_penalty'] *= (1 + random.uniform(-tweak_factor, tweak_factor))
        if 'cohort_bonus' in candidate:
            candidate['cohort_bonus'] *= (1 + random.uniform(-tweak_factor, tweak_factor))
        for key in candidate['keyword_weights']:
            candidate['keyword_weights'][key] *= (1 + random.uniform(-tweak_factor, tweak_factor))

        target_embedding = get_target_embedding(candidate)
        score_zero = CSVgrader.calculate_score(error_entry['zero_abstract'], candidate, target_embedding)
        score_one = CSVgrader.calculate_score(error_entry['one_abstract'], candidate, target_embedding)
        candidate_error = score_zero - score_one

        if candidate_error < best_error:
            best_error = candidate_error
            best_config = candidate
            if best_error <= 0:
                break
    return best_config

# ------------------------
# ERROR PROCESSING
# ------------------------
def process_error(error_entry, config, included_abstracts, non_included_abstracts,
                  inner_patience=3, master_patience=5, monte_carlo_iterations=50, tweak_factor=0.1):
    """
    Process a single error by repeatedly attempting a Monte Carlo tweak to reduce the error.
    If improvement is not achieved within inner_patience iterations, add new keywords sparingly.
    The loop stops once the error is fixed (error difference <= 0) or master_patience is exceeded.
    """
    error_cutoff = 0
    master_counter = 0
    current_error = error_entry['zero_score'] - error_entry['one_score']

    while current_error > error_cutoff and master_counter < master_patience:
        print("\n--- Processing an error with advanced loop ---")
        print("Initial error for this pair:", current_error)
        inner_iter = 0
        improved = False

        # Monte Carlo simulation loop
        while inner_iter < inner_patience:
            candidate_config = monte_carlo_tweak(error_entry, config, max_iterations=monte_carlo_iterations, tweak_factor=tweak_factor)
            target_embedding = get_target_embedding(candidate_config)
            new_zero_score = CSVgrader.calculate_score(error_entry['zero_abstract'], candidate_config, target_embedding)
            new_one_score = CSVgrader.calculate_score(error_entry['one_abstract'], candidate_config, target_embedding)
            new_error = new_zero_score - new_one_score

            # Check if new error is significantly better than current
            error_difference = current_error - new_error

            print(f"Monte Carlo iteration {inner_iter+1}: error correction = {error_difference:.3f}")

            if error_difference > error_cutoff:
                # Update config and error if improvement is achieved
                config = candidate_config
                current_error = new_error
                improved = True
                if current_error <= error_cutoff:
                    print("Error fixed in Monte Carlo loop!")
                    break
            inner_iter += 1

        if current_error <= error_cutoff:
            break  # Error fixed successfully

        # If no improvement in the Monte Carlo loop, add new keywords sparingly.
        if not improved:
            print("No sufficient improvement; adding new keywords...")
            # I would like it to add only keywords that it has positively found in many entries
            config = add_new_keywords_sparingly(error_entry, config, included_abstracts, non_included_abstracts)
            # Optionally, update the error scores after keyword addition.
            target_embedding = get_target_embedding(config)
            current_error = CSVgrader.calculate_score(error_entry['zero_abstract'], config, target_embedding) - \
                            CSVgrader.calculate_score(error_entry['one_abstract'], config, target_embedding)
            print("New error after adding keywords:", current_error)
            master_counter += 1

    if master_counter >= master_patience:
        print("Master patience exceeded for this error.")
    return config

# ------------------------
# MAIN PIPELINE
# ------------------------
def pipeline_tuning(input_file_path, output_file_path, max_outer_iterations=20, monte_carlo_iterations=50,
                    base_tweak_factor=0.1, patience=3, master_patience=5, delimiter=";"):
    config = load_config('config.json')
    df = pd.read_csv(input_file_path, delimiter=delimiter)
    df = df.dropna(subset=['INCLUDED'])
    df['INCLUDED'] = df['INCLUDED'].astype(int)

    # Declaring global included and non-included abstracts
    global all_included_abstracts, all_non_included_abstracts
    all_included_abstracts = [str(x) for x in df[df['INCLUDED'] == 1]['ABSTRACT'].tolist()]
    all_non_included_abstracts = [str(x) for x in df[df['INCLUDED'] == 0]['ABSTRACT'].tolist()]

    df = compute_scores(df, config)
    current_separation = check_separation(df, 'SCORE', 'INCLUDED')
    print(f"Initial separation check: {current_separation}")

    for outer_iter in range(1, max_outer_iterations + 1):
        print(f"\n=== Outer Iteration {outer_iter} ===")
        df = compute_scores(df, config)
        current_separation = check_separation(df, 'SCORE', 'INCLUDED')
        print(f"Separation check: {current_separation}")

        # Breaks loop if separation is achieved
        if current_separation:
            print("Perfect separation achieved!")
            break

        errors = identify_errors(df)
        print(f"Found {len(errors)} error(s).")

        # Process each error using the new advanced loop.
        tqdm.pandas(desc="Correcting errors")
        for error in tqdm(errors):
            # Process each error individually
            config = process_error(error, config, all_included_abstracts, all_non_included_abstracts,
                                     inner_patience=patience, master_patience=master_patience,
                                     monte_carlo_iterations=monte_carlo_iterations, tweak_factor=base_tweak_factor)
            # Update error scores based on the new configuration
            target_embedding = get_target_embedding(config)
            error['zero_score'] = CSVgrader.calculate_score(error['zero_abstract'], config, target_embedding)
            error['one_score'] = CSVgrader.calculate_score(error['one_abstract'], config, target_embedding)
            print("Post-process error for this pair:", error['zero_score'] - error['one_score'])

        df = compute_scores(df, config)
        if check_separation(df, 'SCORE', 'INCLUDED'):
            print("Perfect separation achieved!")
            break

    save_config(config, 'config.json')
    df.to_csv(output_file_path, index=False)
    print("\nFinal configuration saved to config.json")
    print(f"Updated graded articles saved to '{output_file_path}'.")
    return config

if __name__ == '__main__':
    print("LOL")

    # Example for calling whole process

    # final_config = pipeline_tuning(
    #    input_file_path='Articles/Scored_articles/scored_graded_articles1.csv',
    #    output_file_path='graded_articles_updated.csv',
    #    max_outer_iterations=20,
    #    monte_carlo_iterations=50,
    #    base_tweak_factor=0.1,
    #    patience=3
    #)