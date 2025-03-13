import pandas as pd
import spacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib

# ✅ Step 1: Load Data
def load_data(file_path):
    print("🚀 Loading data...")
    df = pd.read_csv(file_path)
    
    # Ensure 'entities' is converted from string to dictionary
    df['entities'] = df['entities'].apply(eval)
    
    print(f"✅ Data loaded: {len(df)} records")
    return df

# ✅ Step 2: Extract Features
nlp = spacy.load("en_core_web_sm")

def extract_features(doc, i):
    token = doc[i]
    features = {
        'bias': 1.0,
        'word.lower()': token.text.lower(),
        'word.isupper()': token.is_upper,
        'word.istitle()': token.is_title,
        'word.isdigit()': token.is_digit,
        'pos': token.pos_,
        'shape': token.shape_,
    }

    if i > 0:
        token_prev = doc[i - 1]
        features.update({
            '-1:word.lower()': token_prev.text.lower(),
            '-1:pos': token_prev.pos_,
            '-1:shape': token_prev.shape_,
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(doc) - 1:
        token_next = doc[i + 1]
        features.update({
            '+1:word.lower()': token_next.text.lower(),
            '+1:pos': token_next.pos_,
            '+1:shape': token_next.shape_,
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

def extract_features_from_sentence(sentence):
    doc = nlp(sentence)
    return [extract_features(doc, i) for i in range(len(doc))]

def extract_labels_from_entities(tokens, entities):
    labels = ["O"] * len(tokens)
    for start, end, label in entities:
        for i, token in enumerate(tokens):
            if token.idx == start:
                labels[i] = f"B-{label}"
            elif token.idx > start and token.idx < end:
                labels[i] = f"I-{label}"
    return labels

def prepare_data(df):
    X, y = [], []
    for _, row in df.iterrows():
        text = row['text']
        entities = row['entities']['entities']
        doc = nlp(text)
        tokens = [token for token in doc]
        X.append(extract_features_from_sentence(text))
        y.append(extract_labels_from_entities(tokens, entities))
    return X, y

# ✅ Step 3: Split Dataset
def split_data(X, y):
    print("\n✂️ Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Training size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# ✅ Step 4: Train CRF Model
def train_crf(X_train, y_train):
    print("\n⚙️ Training CRF model...")
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("✅ Model trained!")
    return crf

# ✅ Step 5: Evaluate Model
def evaluate_model(crf, X_test, y_test):
    print("\n📊 Evaluating model...")
    y_pred = crf.predict(X_test)
    report = flat_classification_report(y_test, y_pred)
    print(report)

# ✅ Step 6: Optimize Hyperparameters
def optimize_crf(X_train, y_train):
    print("\n🔍 Tuning hyperparameters...")
    params_space = {
        'c1': [0.1, 0.2, 0.5],
        'c2': [0.1, 0.2, 0.5],
    }

    crf = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )

    search = GridSearchCV(crf, params_space, cv=3, verbose=1)
    search.fit(X_train, y_train)

    print(f"\n✅ Best params: {search.best_params_}")
    return search.best_estimator_

# ✅ Step 7: Save Model
def save_model(model, filename="ner_crf_model.pkl"):
    joblib.dump(model, filename)
    print(f"\n✅ Model saved as '{filename}'")

# ✅ Step 8: Load Model
def load_model(filename="ner_crf_model.pkl"):
    model = joblib.load(filename)
    print("\n✅ Model loaded!")
    return model

# ✅ Step 9: Main Function
def main():
    # Load data
    df = load_data("data/preprocessed_data.csv")
    
    # Extract features and labels
    print("\n📑 Extracting features and labels...")
    X, y = prepare_data(df)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train CRF Model
    crf = train_crf(X_train, y_train)

    # Evaluate Model
    evaluate_model(crf, X_test, y_test)

    # Optimize Model
    crf = optimize_crf(X_train, y_train)

    # Save Model
    save_model(crf)

    # Load Model and Test Again
    crf = load_model()
    evaluate_model(crf, X_test, y_test)

if __name__ == "__main__":
    main()
