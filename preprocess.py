import pandas as pd
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
import ast

# ✅ Step 1: Load Dataset
def load_data():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset

# ✅ Step 2: Clean Entities
def clean_entities(row):
    try:
        # Convert string to dictionary
        entities = ast.literal_eval(row['entities'])  
        cleaned_entities = []
        for ent in entities['entities']:
            if len(ent) == 3 and isinstance(ent[0], int) and isinstance(ent[1], int) and isinstance(ent[2], str):
                # Fix double dashes and empty labels
                ent = (ent[0], ent[1], ent[2].replace('--', '-'))
                cleaned_entities.append(ent)
        
        if cleaned_entities:
            return {'entities': cleaned_entities}
        else:
            return None  # Remove row if no valid entities remain
    except Exception as e:
        return None  # Remove row if parsing fails

# ✅ Step 3: Exploratory Data Analysis (EDA)
def perform_eda(dataset):
    label_map = dataset['train'].features['ner_tags'].feature

    # ➡️ NER Tag Distribution
    all_labels = [label for example in dataset['train']['ner_tags'] for label in example]
    decoded_labels = [label_map.int2str(label) for label in all_labels]
    label_counts = pd.Series(decoded_labels).value_counts()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("NER Tag Distribution")
    plt.show()

    # ➡️ Sentence Length Distribution
    sentence_lengths = [len(example) for example in dataset['train']['tokens']]
    plt.figure(figsize=(8, 5))
    sns.histplot(sentence_lengths, bins=20, kde=True, color='skyblue')
    plt.title("Sentence Length Distribution")
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.show()

    # ➡️ Most Frequent Tokens
    tokens = [token.lower() for example in dataset['train']['tokens'] for token in example]
    token_counts = Counter(tokens)
    top_tokens = token_counts.most_common(20)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=[token[0] for token in top_tokens], y=[token[1] for token in top_tokens])
    plt.xticks(rotation=45)
    plt.title("Top 20 Most Frequent Tokens")
    plt.show()

# ✅ Step 4: Preprocessing
def preprocess_data(dataset):
    nlp = spacy.blank("en")  # Create blank spaCy model
    
    # Add NER pipeline if not exists
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Load label map
    label_map = dataset['train'].features['ner_tags'].feature

    train_data = []
    for example in dataset['train']:
        words = example['tokens']
        entities = example['ner_tags']
        entity_offsets = []
        offset = 0

        for word, entity in zip(words, entities):
            start = offset
            end = offset + len(word)
            entity_label = label_map.int2str(entity)
            if entity_label != 'O':
                entity_offsets.append((start, end, entity_label))
            offset = end + 1

        text = " ".join(words)
        train_data.append((text, {"entities": entity_offsets}))

    # Add labels to NER pipeline
    for _, annotations in train_data:
        for start, end, label in annotations['entities']:
            ner.add_label(label)

    return nlp, train_data

# ✅ Step 5: Save Preprocessed Data
def save_preprocessed_data(train_data):
    os.makedirs("data", exist_ok=True)  # Create 'data' folder if it doesn't exist
    df = pd.DataFrame(train_data, columns=["text", "entities"])
    df.to_csv("data/preprocessed_data.csv", index=False)
    print("\n✅ Preprocessed data saved to 'data/preprocessed_data.csv'")

# ✅ Step 6: Main Function
def main():
    print("\n🚀 Loading dataset...")
    dataset = load_data()

    # ➡️ Convert dataset to DataFrame format for cleaning
    label_map = dataset['train'].features['ner_tags'].feature
    df = pd.DataFrame({
        'text': [" ".join(example['tokens']) for example in dataset['train']],
        'entities': [str({
            'entities': [
                (start, end, label_map.int2str(entity))
                for start, end, entity in zip(
                    range(len(example['tokens'])),
                    range(1, len(example['tokens']) + 1),
                    example['ner_tags']
                )
            ]
        }) for example in dataset['train']]
    })

    print("\n🧹 Cleaning entities...")
    df['entities'] = df['entities'].apply(clean_entities)
    df.dropna(subset=['entities'], inplace=True)

    print(f"✅ Cleaned dataset. Remaining rows: {len(df)}")

    print("\n🔎 Performing EDA...")
    perform_eda(dataset)

    print("\n⚙️ Preprocessing data...")
    nlp, train_data = preprocess_data(dataset)

    print("\n💾 Saving preprocessed data...")
    save_preprocessed_data(train_data)

if __name__ == "__main__":
    main()
