import csv
import os
import re

def clean_text(text):
    # Remove newlines and extra spaces
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_file, output_file):
    print(f"Reading from {input_file}...")
    
    pairs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'input' in row and 'output' in row:
                inp = clean_text(row['input'])
                out = clean_text(row['output'])
                if inp and out:
                    # Combine input and output with a separator or just space
                    # For causal LM, we just need a stream of text usually, 
                    # but for conversation, maybe we want "Input: ... Output: ..."?
                    # The user said "Merge the input and output texts into a single sequence for training (for conversational context)"
                    # Simple contactenation "input output" is efficient for general LM.
                    pairs.append(f"{inp} {out}")
    
    print(f"Found {len(pairs)} conversation pairs.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(pair + '\n')
            
    print("Preprocessing complete!")

if __name__ == "__main__":
    input_csv = "hinglish_conversations.csv"
    output_txt = "data/cleaned_hinglish.txt"
    
    if os.path.exists(input_csv):
        preprocess_data(input_csv, output_txt)
    else:
        print(f"Error: {input_csv} not found!")
