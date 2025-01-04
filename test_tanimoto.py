from lib.utils import *

exit()
# Example molecules
mol1 = Chem.MolFromSmiles('CBr')
mol2 = Chem.MolFromSmiles('CC')

# Raw weights for atoms (e.g., frequencies or other importance measures)
raw_weights = {"C": 0.9, "Br": 0.0003}

# Compute weights using softmax
weights = compute_weights_with_softmax(raw_weights) if raw_weights else None

# Calculate similarity with weighting based on atom weights
similarity = calculate_mcs_similarity(mol1, mol2, weights=weights)

# Output
print("Computed weights:", weights)
print("Similarity:", similarity)

print()
