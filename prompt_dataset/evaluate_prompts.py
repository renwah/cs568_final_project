import spacy
import textdescriptives as td
import pandas as pd
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/all")
# Load the dataset
file_path = "/Users/renwah/Downloads/archive/processed_prompt_examples_dataset.csv"
df = pd.read_csv(file_path)

# Process each prompt_example and extract features
results = []
for prompt in df['prompt_example']:
    doc = nlp(prompt)
    results.append({
        "readability": doc._.readability,
        "token_length": doc._.token_length,
        "sentence_length": doc._.sentence_length,
        "coherence": doc._.coherence,
        "information_theory": doc._.information_theory,
        "entropy": doc._.entropy,
        "perplexity": doc._.perplexity,
        "per_word_perplexity": doc._.per_word_perplexity
    })

# Convert results to a DataFrame and merge with the original dataset
results_df = pd.DataFrame(results)
df = pd.concat([df, results_df], axis=1)

# Save the updated dataset
df.to_csv(file_path, index=False)