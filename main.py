# ---- Hyperparameters ----
EPOCHS = 3
BATCH_SIZE = 32
SPLIT_RATIO = 0.8
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

# ---- Imports ----
import os
import glob
import random
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# ---- Download and extract data ----
if os.path.exists("data"):
    print("Data directory already exists, skipping download and extraction.")
else:
    url = "https://github.com/JeremyTubongbanua/sofe4620u-nba/raw/trunk/data.zip"

    response = requests.get(url)
    if response.status_code == 200:
        with open("data.zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall()

        extracted_files = []
        for root, dirs, files in os.walk("data"):
            for file in files:
                extracted_files.append(os.path.join(root, file))

        print('Downloaded:')
        for file in extracted_files:
            print(f"- {file}")

        print(f"\nTotal files extracted: {len(extracted_files)}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# ---- Read data -----

with open("data/games.txt", "r") as f:
    games = {line.strip(): i for i, line in enumerate(f.readlines())}

with open("data/names.txt", "r") as f:
    names = [line.strip() for line in f.readlines()]

with open("data/seasons.txt", "r") as f:
    seasons = {line.strip(): i for i, line in enumerate(f.readlines())}

with open("data/teams.txt", "r") as f:
    teams = [line.strip() for line in f.readlines()]

# ---- Set up dataframe ----

columns = [
    "game", "season", "home_team", "away_team", "starting_min", "outcome",
    "home_players", "away_players"
]

df = pd.read_csv("data/mapped_data.csv", usecols=columns)

# converts string of list of players indices to list of integers
for i in range(len(df)):
    away_team_list_of_players_str = df.iloc[i]['away_players']
    away_team_list_of_players = list(map(int, away_team_list_of_players_str[1:-1].split(',')))
    df.at[i, 'away_players'] = away_team_list_of_players
    home_players_list_of_players_str = df.iloc[i]['home_players']
    home_players_list_of_players = list(map(int, home_players_list_of_players_str[1:-1].split(',')))
    df.at[i, 'home_players'] = home_players_list_of_players

# remove all rows with outcome == -1
df_positive = df[df['outcome'] == 1].copy()

# ----- Create new rows by removing one player from each row -----

new_rows = []
for idx, row in df_positive.iterrows():
    home_players = row['home_players'].copy()

    if len(home_players) == 5:
        for player_to_remove in home_players:
            new_row_dict = row.to_dict()
            new_row_dict['home_players'] = [p for p in home_players if p != player_to_remove]
            new_row_dict['player_removed'] = player_to_remove
            new_rows.append(new_row_dict)

df_expanded = pd.DataFrame(new_rows)

print(f"Original dataframe size: {len(df)}")
print(f"Positive outcome dataframe size: {len(df_positive)}")
print(f"Expanded dataframe size: {len(df_expanded)}")
print(df_expanded.head())

class BasketballTeamGNN(nn.Module):
    def __init__(self, num_players, embedding_dim, hidden_dim):
        super(BasketballTeamGNN, self).__init__()
        
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        
        self.home_conv1 = GCNConv(embedding_dim, hidden_dim)
        self.home_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.away_conv1 = GCNConv(embedding_dim, hidden_dim)
        self.away_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.team_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.player_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_players)
        )
    
    def forward(self, home_data, away_data, context):
        home_x = self.player_embedding(home_data.x)
        home_x = F.relu(self.home_conv1(home_x, home_data.edge_index))
        home_x = F.dropout(home_x, p=0.2, training=self.training)
        home_x = self.home_conv2(home_x, home_data.edge_index)
        home_team_embedding = global_mean_pool(home_x, home_data.batch)

        away_x = self.player_embedding(away_data.x)
        away_x = F.relu(self.away_conv1(away_x, away_data.edge_index))
        away_x = F.dropout(away_x, p=0.2, training=self.training)
        away_x = self.away_conv2(away_x, away_data.edge_index)
        away_team_embedding = global_mean_pool(away_x, away_data.batch)

        context_embedding = self.context_encoder(context)

        team_diff = home_team_embedding - away_team_embedding
        team_features = torch.cat([home_team_embedding, away_team_embedding, context_embedding], dim=1)
        team_encoding = self.team_encoder(team_features)

        player_logits = self.player_predictor(torch.cat([team_encoding, team_diff], dim=1))

        return player_logits


class BasketballDataset(Dataset):
    def __init__(self, dataframe, num_players):
        self.df = dataframe
        self.num_players = num_players
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        home_players = torch.tensor(row['home_players'], dtype=torch.long)
        away_players = torch.tensor(row['away_players'], dtype=torch.long)
        player_removed = torch.tensor(row['player_removed'], dtype=torch.long)
        
        home_team_size = len(home_players)
        away_team_size = len(away_players)
        
        home_edges = []
        for i in range(home_team_size):
            for j in range(home_team_size):
                if i != j:
                    home_edges.append([i, j])
        home_edge_index = torch.tensor(home_edges, dtype=torch.long).t().contiguous()
        
        away_edges = []
        for i in range(away_team_size):
            for j in range(away_team_size):
                if i != j:
                    away_edges.append([i, j])
        away_edge_index = torch.tensor(away_edges, dtype=torch.long).t().contiguous()
        
        home_data = Data(x=home_players, edge_index=home_edge_index)
        away_data = Data(x=away_players, edge_index=away_edge_index)
        
        context = torch.tensor([row['season'], row['starting_min']], dtype=torch.float)
        
        return home_data, away_data, context, player_removed


def collate_fn(batch):
    home_data_list, away_data_list, contexts, players_removed = zip(*batch)
    
    home_batch = Batch.from_data_list(home_data_list)
    away_batch = Batch.from_data_list(away_data_list)
    
    contexts = torch.stack(contexts)
    players_removed = torch.stack(players_removed)
    
    return home_batch, away_batch, contexts, players_removed


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for home_data, away_data, context, target in train_loader:
            # Move data to device
            home_data = home_data.to(device)
            away_data = away_data.to(device)
            context = context.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(home_data, away_data, context)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            epoch_train_loss += loss.item() * target.size(0)
        
        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / train_total
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for home_data, away_data, context, target in val_loader:
                # Move data to device
                home_data = home_data.to(device)
                away_data = away_data.to(device)
                context = context.to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(home_data, away_data, context)
                
                # Calculate loss
                loss = criterion(output, target)
                epoch_val_loss += loss.item() * target.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate epoch metrics
        avg_val_loss = epoch_val_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, train_accs, val_losses, val_accs


# Get all unique players to determine num_players
all_players = set()
for players in df_expanded['home_players']:
    all_players.update(players)
all_players.update(df_expanded['player_removed'])
num_players = max(all_players) + 1

# Split into train and test sets
train_df, val_df = train_test_split(df_expanded, test_size=1-SPLIT_RATIO, random_state=42)

# Create datasets and dataloaders
train_dataset = BasketballDataset(train_df, num_players)
val_dataset = BasketballDataset(val_df, num_players)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
model = BasketballTeamGNN(num_players, EMBEDDING_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
train_losses, train_accs, val_losses, val_accs = train_model(
    model, train_loader, val_loader, optimizer, criterion, EPOCHS
)

# Save the trained model
torch.save(model.state_dict(), 'basketball_player_prediction_model.pt')

# Plot training and validation metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Print final metrics
print("\nFinal Results:")
print(f"Training Accuracy: {train_accs[-1]:.4f}")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Validation Accuracy: {val_accs[-1]:.4f}")
print(f"Validation Loss: {val_losses[-1]:.4f}")

# Additional model evaluation
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Evaluate top-K accuracy on validation set
k_values = [1, 3, 5, 10]
top_k_correct = {k: 0 for k in k_values}
total = 0

with torch.no_grad():
    for home_data, away_data, context, target in val_loader:
        home_data = home_data.to(device)
        away_data = away_data.to(device)
        context = context.to(device)
        target = target.to(device)
        
        output = model(home_data, away_data, context)
        
        # Calculate top-k accuracy
        _, top_k_indices = torch.topk(output, max(k_values), dim=1)
        
        for k in k_values:
            top_k_indices_k = top_k_indices[:, :k]
            batch_correct = torch.any(top_k_indices_k == target.unsqueeze(1), dim=1).sum().item()
            top_k_correct[k] += batch_correct
        
        total += target.size(0)

# Print top-K accuracy results
print("\nTop-K Accuracy:")
for k in k_values:
    print(f"Top-{k} Accuracy: {top_k_correct[k]/total:.4f}")