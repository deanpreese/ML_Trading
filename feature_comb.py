import itertools


features = [
    'RSI',
    'STOK1',
    'SDKC9',
    'SDLR310',
    'ATR2',
    'SDKC91',
    'SDBB91',
    'ATR3',
    'ATR21',
    'ATR31',
    'ROC',
    'ATR32',
    'ATR34'
]

# Function to generate combinations
def generate_combinations(features, min_features=3, max_features=7):
    all_combinations = []
    for r in range(min_features, max_features + 1):
        combinations = list(itertools.combinations(features, r))
        all_combinations.extend(combinations)
    return all_combinations

# Generate combinations
combinations = generate_combinations(features)

# Print the combinations
for combination in combinations:
    print(combination)

# Save combinations to a file
#with open('feature_combinations.txt', 'w') as f:
#    for combination in combinations:
#        f.write(f"{combination}\n")