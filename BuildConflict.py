import numpy as np
import pandas as pd
import random

# List of first names and last names
first_names = ["Ali", "Zahra", "Reza", "Sara", "Mohammad", "Fatemeh", "Hossein", "Maryam", "Mehdi", "Narges", "Hamed", "Roya"]
last_names = ["Ahmadi", "Hosseini", "Karimi", "Rahimi", "Hashemi", "Ebrahimi", "Moradi", "Mohammadi", "Rostami", "Fazeli", "Hosseinzadeh", "Niknam"]

# Set a seed for reproducibility
random.seed(42)

# Generate 24 unique random names
random_names = set()
while len(random_names) < 24:
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    random_names.add(f"{first_name} {last_name}")

random_names = list(random_names)  # Convert set to list

# Initialize a 24x24 matrix with zeros
conflict_matrix = np.zeros((24, 24), dtype=int)

# Randomly select 40 unique pairs for conflicts
conflict_pairs = set()
while len(conflict_pairs) < 40:
    i = random.randint(0, 23)
    j = random.randint(0, 23)
    if i != j and (i, j) not in conflict_pairs:
        conflict_pairs.add((i, j))

# Assign conflicts to the matrix
for i, j in conflict_pairs:
    conflict_matrix[i, j] = 1

# Create a DataFrame with the conflict matrix
conflict_df = pd.DataFrame(conflict_matrix, index=random_names, columns=random_names)

# Display the DataFrame
print(conflict_df)
