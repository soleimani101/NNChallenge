import torch
import torch.nn as nn
import torch.optim as optim


# Function to calculate the conflict score for a seating arrangement
def calculate_conflict_score(seating, conflict_matrix):
    conflict_score = 0
    for i in range(24):
        for j in range(24):
            if conflict_matrix[seating[i], seating[j]] == 1:
                if i != j and (abs(i // 4 - j // 4) <= 1 and abs(i % 4 - j % 4) <= 1):
                    conflict_score += 1
    return torch.tensor(conflict_score, dtype=torch.float32)


# Define the neural network architecture
class SeatingNet(nn.Module):
    def __init__(self):
        super(SeatingNet, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the neural network
model = SeatingNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Set the conflicts between people
def generate_conflict_matrix(num_people, num_conflicts):
    conflict_matrix = torch.zeros((num_people, num_people))

    # Generate random conflicts
    for _ in range(num_conflicts):
        while True:
            person1 = torch.randint(0, num_people, (1,)).item()
            person2 = torch.randint(0, num_people, (1,)).item()

            # Ensure the conflict is between different people and not already assigned
            if person1 != person2 and conflict_matrix[person1, person2] == 0:
                conflict_matrix[person1, person2] = 1
                conflict_matrix[person2, person1] = 1
                break

    return conflict_matrix


conflict_matrix = generate_conflict_matrix(24, 40)
# Add more conflicts as needed

# Print the conflicts between people
print("Conflicts between people:")
for i in range(24):
    for j in range(i + 1, 24):
        if conflict_matrix[i, j] == 1:
            print(f"Person {i} has a conflict with Person {j}")

# Generate a dataset of seating arrangements and their conflict scores
dataset = []
for _ in range(1000):
    # Generate a random seating arrangement
    seating = torch.randperm(24)
    # Calculate the conflict score for the seating arrangement
    conflict_score = calculate_conflict_score(seating, conflict_matrix)
    dataset.append((seating, conflict_score))

# Train the neural network
num_epochs = 100
for epoch in range(num_epochs):
    for seating, conflict_score in dataset:
        # Forward pass
        output = model(seating.float())
        loss = criterion(output, conflict_score.float().unsqueeze(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Use the trained model to find the optimal seating arrangement
optimal_seating = torch.randperm(24)
optimal_conflict_score = float('inf')

for _ in range(1000):
    seating = torch.randperm(24)
    conflict_score = model(seating.float()).item()
    if conflict_score < optimal_conflict_score:
        optimal_seating = seating
        optimal_conflict_score = conflict_score

# Reshape the optimal seating arrangement into a 6x4 matrix
optimal_seating_matrix = optimal_seating.view(6, 4)
print("Optimal Seating Arrangement:")
print(optimal_seating_matrix)

# Count the total number of conflicts in the optimal seating arrangement
total_conflicts = calculate_conflict_score(optimal_seating, conflict_matrix).item()
print(f"Total conflicts in the optimal seating arrangement: {total_conflicts}")
