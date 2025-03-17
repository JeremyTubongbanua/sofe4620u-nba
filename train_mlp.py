# ---- Hyperparameters ----
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

# ---- Imports ----
import os
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from tqdm import tqdm

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

class BasketballMLP(nn.Module):
    def __init__(self, num_players, embedding_dim, hidden_dim):
        super(BasketballMLP, self).__init__()
        
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        
        self.context_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim + 5 * embedding_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_players)
        )
    
    def forward(self, home_players, away_players, context):
        home_embeddings = self.player_embedding(home_players)
        home_embeddings_flat = home_embeddings.view(home_embeddings.size(0), -1)
        
        away_embeddings = self.player_embedding(away_players)
        away_embeddings_flat = away_embeddings.view(away_embeddings.size(0), -1)
        context_features = self.context_encoder(context)
        
        combined_features = torch.cat([home_embeddings_flat, away_embeddings_flat, context_features], dim=1)
        
        logits = self.mlp(combined_features)
        
        return logits


class BasketballMLPDataset(Dataset):
    def __init__(self, dataframe, num_players):
        self.df = dataframe
        self.num_players = num_players
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        home_players = row['home_players']
        home_players_padded = home_players + [0] * (4 - len(home_players))
        home_players_tensor = torch.tensor(home_players_padded, dtype=torch.long)
        
        away_players = row['away_players']
        away_players_padded = away_players + [0] * (5 - len(away_players))
        away_players_tensor = torch.tensor(away_players_padded, dtype=torch.long)
        
        player_removed = torch.tensor(row['player_removed'], dtype=torch.long)
        
        context = torch.tensor([row['season'], row['starting_min']], dtype=torch.float)
        
        return home_players_tensor, away_players_tensor, context, player_removed


def train_mlp_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for home_players, away_players, context, target in tqdm(train_loader, desc="Training", leave=False):
            home_players = home_players.to(device)
            away_players = away_players.to(device)
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(home_players, away_players, context)
            
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            epoch_train_loss += loss.item() * target.size(0)
        
        avg_train_loss = epoch_train_loss / train_total
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        
        model.eval()
        epoch_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for home_players, away_players, context, target in tqdm(val_loader, desc="Validation", leave=False):
                home_players = home_players.to(device)
                away_players = away_players.to(device)
                context = context.to(device)
                target = target.to(device)
                
                output = model(home_players, away_players, context)
                
                loss = criterion(output, target)
                epoch_val_loss += loss.item() * target.size(0)
                
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        avg_val_loss = epoch_val_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)
        
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, train_accs, val_losses, val_accs


def evaluate_topk_accuracy(model, data_loader, k_values=[1, 3, 5, 10]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    top_k_correct = {k: 0 for k in k_values}
    total = 0
    
    with torch.no_grad():
        for home_players, away_players, context, target in tqdm(data_loader, desc="Evaluating"):
            home_players = home_players.to(device)
            away_players = away_players.to(device)
            context = context.to(device)
            target = target.to(device)
            
            output = model(home_players, away_players, context)
            
            _, top_k_indices = torch.topk(output, max(k_values), dim=1)
            
            for k in k_values:
                top_k_indices_k = top_k_indices[:, :k]
                batch_correct = torch.any(top_k_indices_k == target.unsqueeze(1), dim=1).sum().item()
                top_k_correct[k] += batch_correct
            
            total += target.size(0)
    
    top_k_accuracies = {k: top_k_correct[k]/total for k in k_values}
    
    return top_k_accuracies


SPLIT_RATIO = 0.8

all_players = set()
for players in df_expanded['home_players']:
    all_players.update(players)
all_players.update(df_expanded['player_removed'])
num_players = max(all_players) + 1

train_df, val_df = train_test_split(df_expanded, test_size=1-SPLIT_RATIO, random_state=42)

train_dataset = BasketballMLPDataset(train_df, num_players)
val_dataset = BasketballMLPDataset(val_df, num_players)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = BasketballMLP(num_players, EMBEDDING_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, train_accs, val_losses, val_accs = train_mlp_model(
    model, train_loader, val_loader, optimizer, criterion, EPOCHS
)

torch.save(model.state_dict(), 'basketball_mlp_model.pt')

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
plt.savefig('mlp_training_metrics.png')
plt.show()

print("\nFinal Results:")
print(f"Training Accuracy: {train_accs[-1]:.4f}")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Validation Accuracy: {val_accs[-1]:.4f}")
print(f"Validation Loss: {val_losses[-1]:.4f}")

topk_results = evaluate_topk_accuracy(model, val_loader)
print("\nTop-K Accuracy:")
for k, acc in topk_results.items():
    print(f"Top-{k} Accuracy: {acc:.4f}")