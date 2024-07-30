import pandas as pd
from transformers import pipeline, GPT2Tokenizer
import torch

datafile = [ 
        'data/buildSeqInd_Lucky13_5M_3070.csv',   #0
        'data/buildSeqInd_Lucky13_5M_ALL.csv',  #1
        'data/buildSeqInd_Lucky13_F.csv',  #2
        'data/buildSeqInd_Lucky13_D.csv',  #3
        'data/buildSeqInd_Lucky13_F_3070.csv',  #4
        'data/Expanded_Lucky13_070.csv',  #5
        'data/alt_ex13x.csv', #6
        'data/ndata_all.csv', #7
        'data/ndata_3070.csv', #8
        'data/lucky13_llm.csv', #9
        'data/ndata_3070_oos.csv', #10
    ]

data = pd.read_csv(datafile[10])
# Select features (assuming your features are all columns except 'target' and 'classification_target')
features = data.drop(columns=['output', 'outputC'])
print(f"Selected features with shape: {features.shape}")

# Prepare feature data as text for LLM
print("Preparing feature data as text for LLM...")
feature_text_list = []
for column in features.columns:
    feature_text_list.append(f"{column}: {features[column].tolist()[:10]}")  # Taking the first 10 values for brevity
feature_text = "\n".join(feature_text_list)
print(f"Feature data prepared with length: {len(feature_text)} characters")

# Initialize the tokenizer and the LLM pipeline with a publicly available model
print("Initializing the LLM pipeline...")
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Check if MPS is available and use it if possible
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

nlp = pipeline('text-generation', model='distilgpt2', device=device)
print("LLM pipeline initialized with model 'distilgpt2'.")

# Maximum sequence length for the model
max_length = tokenizer.model_max_length

# Function to truncate text to fit within model's token limit
def truncate_text(text, max_length):
    encoded_text = tokenizer.encode(text)
    if len(encoded_text) > max_length:
        truncated_text = tokenizer.decode(encoded_text[:max_length])
        print(f"Text truncated to {max_length} tokens.")
    else:
        truncated_text = text
    return truncated_text

# Split the feature text into manageable chunks
def split_into_chunks(text, chunk_size):
    encoded_text = tokenizer.encode(text)
    chunks = [encoded_text[i:i + chunk_size] for i in range(0, len(encoded_text), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

chunk_size = max_length // 2  # Use half the max length for each chunk
chunks = split_into_chunks(feature_text, chunk_size)

# Function to analyze features with LLM
def analyze_features_with_llm(chunks, target_description):
    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        prompt = f"Analyze the following features and their relation to the target variable:\n\n{chunk}\n\nBased on the analysis, suggest the most relevant features for predicting the target. Provide reasons for your suggestions."
        prompt = truncate_text(prompt, max_length)  # Ensure the prompt is within the token limit
        print(f"Prompt created with length: {len(prompt)} characters. Calling LLM...")
        response = nlp(prompt, max_new_tokens=150)
        print("LLM response received.")
        results.append(response[0]['generated_text'])
    return results

# Describe the target variable
target_description = "The target variable represents ..."

# Get LLM analysis and suggestions
print("Analyzing features with LLM...")
llm_responses = analyze_features_with_llm(chunks, target_description)
print("LLM analysis complete.")

# Display LLM responses
for i, response in enumerate(llm_responses):
    print(f"LLM Response for chunk {i+1}:\n", response)

# Verbose: End of the script
print("Script completed.")


"""
Analyze the following features and their relation to the target variable:
'data/lucky13_llm.csv', #9

SDLR310: [0.000201, 0.00023, -0.000147, 1.4e-05, 0.00013, 0.000303, 0.000151, -6.4e-05, -8.2e-05, -0.000148]
SDBB91: [0.000648, 0.000896, 0.001241, 0.001228, 0.000664, 0.000906, 0.001127, 0.001274, 0.001168, 0.001189]
SDKC91: [0.001498, 0.001667, 0.001853, 0.002003, 0.00129, 0.001271, 0.001383, 0.001383, 0.00129, 0.001177]
SDKC9: [0.001667, 0.001779, 0.002003, 0.002115, 0.001271, 0.001383, 0.001383, 0.00129, 0.001177, 0.001159]
ROC: [0.0787, 0.0731, 0.09, 0.1687, 0.0898, 0.1291, 0.1179, 0.1291, 0.101, 0.0729]
ATR34: [1.8966, 1.9673, 2.7191, 2.8752, 2.0978, 2.0283, 1.9726, 1.8781, 2.0025, 2.202]
ATR32: [2.2738, 2.7191, 2.8502, 2.6302, 1.9726, 1.8781, 2.0025, 2.202, 2.0616, 1.8493]
ATR31: [2.7191, 2.8752, 2.6302, 2.7041, 1.8781, 2.0025, 2.202, 2.0616, 1.8493, 1.6794]
ATR3: [2.8752, 2.8502, 2.7041, 2.9133, 2.0025, 2.202, 2.0616, 1.8493, 1.6794, 1.6435]

LLM Response for chunk 2:
Analyze the following features and their relation to the target variable:

ATR21: [3.7126, 3.6063, 2.4641, 2.732, 1.658, 2.079, 2.5395, 2.0198, 1.5099, 1.2549]
ATR2: [3.6063, 3.1782, 2.732, 3.241, 2.079, 2.5395, 2.0198, 1.5099, 1.2549, 1.3775]
RSI: [69.6745, 70.6087, 67.5852, 76.5506, 69.3259, 73.0044, 73.7137, 71.5974, 73.3208, 70.9009]
STOK1: [79.565, 81.8658, 87.2021, 88.7106, 89.4802, 92.6926, 93.0161, 91.8883, 93.8063, 93.8877]
    
    
"""