# ---- Hyperparameters ----
EPOCHS = 3
BATCH_SIZE = 32
SPLIT_RATIO = 0.8
LEARNING_RATE = 0.001

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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    "home_players", "away_players", "player_removed"
]
df = pd.read_csv("data/training_data.csv", usecols=columns)

df_copy = df.copy()

for i in range(len(df_copy)):
    away_team_list_of_players_str = df_copy.iloc[i]['away_players']
    away_team_list_of_players = list(map(int, away_team_list_of_players_str[1:-1].split(',')))
    df_copy.at[i, 'away_players'] = away_team_list_of_players
    home_players_list_of_players_str = df_copy.iloc[i]['home_players']
    home_players_list_of_players = list(map(int, home_players_list_of_players_str[1:-1].split(',')))
    df_copy.at[i, 'home_players'] = home_players_list_of_players

print(df_copy.head())


# ---- Dataset Class ----
class NBADataset(Dataset):
    def __init__(self, dataframe):
        self.game = torch.tensor(dataframe['game'].values, dtype=torch.long)
        self.season = torch.tensor(dataframe['season'].values, dtype=torch.long)
        self.home_team = torch.tensor(dataframe['home_team'].values, dtype=torch.long)
        self.away_team = torch.tensor(dataframe['away_team'].values, dtype=torch.long)
        self.starting_min = torch.tensor(dataframe['starting_min'].values, dtype=torch.long)
        self.outcome = torch.tensor(dataframe['outcome'].values, dtype=torch.float)
        
        home_players_padded = torch.zeros((len(dataframe), 4), dtype=torch.long)
        away_players_padded = torch.zeros((len(dataframe), 5), dtype=torch.long)
        
        for i, (_, row) in enumerate(dataframe.iterrows()):
            home_players = row['home_players']
            away_players = row['away_players']
            
            for j, player in enumerate(home_players):
                home_players_padded[i, j] = player
                
            for j, player in enumerate(away_players):
                away_players_padded[i, j] = player
        
        self.home_players = home_players_padded
        self.away_players = away_players_padded
        self.player_removed = torch.tensor(dataframe['player_removed'].values, dtype=torch.long)
        
    def __len__(self):
        return len(self.game)
    
    def __getitem__(self, idx):
        features = {
            'game': self.game[idx],
            'season': self.season[idx],
            'home_team': self.home_team[idx],
            'away_team': self.away_team[idx],
            'starting_min': self.starting_min[idx],
            'outcome': self.outcome[idx],
            'home_players': self.home_players[idx],
            'away_players': self.away_players[idx]
        }
        
        return features, self.player_removed[idx]

# ---- Model Definition ----
class PlayerPredictionModel(nn.Module):
    def __init__(self, num_games, num_seasons, num_teams, num_players, embedding_dim=32):
        super(PlayerPredictionModel, self).__init__()
        
        self.game_embedding = nn.Embedding(num_games, embedding_dim)
        self.season_embedding = nn.Embedding(num_seasons, embedding_dim)
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.player_embedding = nn.Embedding(num_players + 1, embedding_dim)
        
        # Calculate the correct input size for fc1
        # 1 game + 1 season + 2 teams + 4 home players + 5 away players = 13 embeddings
        # Each embedding is of size embedding_dim
        # Plus 2 additional features: outcome and starting_min
        fc1_input_size = embedding_dim * (1 + 1 + 2 + 4 + 5) + 2
        
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_players)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        game_emb = self.game_embedding(x['game'])
        season_emb = self.season_embedding(x['season'])
        home_team_emb = self.team_embedding(x['home_team'])
        away_team_emb = self.team_embedding(x['away_team'])
        
        home_players_emb = self.player_embedding(x['home_players'])
        away_players_emb = self.player_embedding(x['away_players'])
        
        home_players_emb_flat = home_players_emb.view(home_players_emb.size(0), -1)
        away_players_emb_flat = away_players_emb.view(away_players_emb.size(0), -1)
        
        outcome = x['outcome'].unsqueeze(1)
        starting_min = x['starting_min'].float().unsqueeze(1)
        
        combined = torch.cat([
            game_emb, 
            season_emb, 
            home_team_emb, 
            away_team_emb, 
            home_players_emb_flat, 
            away_players_emb_flat,
            outcome,
            starting_min
        ], dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ---- Data Preparation ----
training_dataframes = df_copy.sample(frac=SPLIT_RATIO, random_state=42)
testing_dataframes = df_copy.drop(training_dataframes.index)

train_dataset = NBADataset(training_dataframes)
test_dataset = NBADataset(testing_dataframes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model Training ----
num_games = len(games)
num_seasons = len(seasons)
num_teams = len(teams)
num_players = len(names)

# Debug - print out dimensions
print(f"Number of games: {num_games}")
print(f"Number of seasons: {len(seasons)}")
print(f"Number of teams: {len(teams)}")
print(f"Number of players: {len(names)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Calculate expected embedding dimension
embedding_dim = 32
expected_input_size = embedding_dim * (1 + 1 + 2 + 4 + 5) + 2  # game, season, 2 teams, 4 home players, 5 away players, outcome, starting_min
print(f"Expected input size to fc1: {expected_input_size}")

model = PlayerPredictionModel(num_games, num_seasons, num_teams, num_players).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for batch_idx, (features, targets) in enumerate(train_loader):
    print("Sample batch shapes:")
    print(f"Game: {features['game'].shape}")
    print(f"Season: {features['season'].shape}")
    print(f"Home team: {features['home_team'].shape}")
    print(f"Away team: {features['away_team'].shape}")
    print(f"Home players: {features['home_players'].shape}")
    print(f"Away players: {features['away_players'].shape}")
    print(f"Outcome: {features['outcome'].shape}")
    print(f"Starting min: {features['starting_min'].shape}")
    print(f"Targets: {targets.shape}")
    
    for k, v in features.items():
        features[k] = v[0:1].to(device)
    
    targets = targets[0:1].to(device)
    
    try:
        outputs = model(features)
        print(f"Forward pass successful. Output shape: {outputs.shape}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
    
    break

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, targets) in enumerate(train_loader):
        # Move data to device
        targets = targets.to(device)
        for k, v in features.items():
            features[k] = v.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {running_loss/(batch_idx+1):.4f}, Acc: {100.*correct/total:.2f}%')

    print(f'Epoch {epoch+1} completed. Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100.*correct/total:.2f}%')

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            # Move data to device
            targets = targets.to(device)
            for k, v in features.items():
                features[k] = v.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {100.*correct/total:.2f}%')

print('Training complete')

torch.save(model.state_dict(), 'nba_player_prediction_model.pth')

def predict_removed_player(model, features):
    model.eval()
    with torch.no_grad():
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.to(device)
            else:
                features[k] = torch.tensor(v, device=device).unsqueeze(0)

        output = model(features)
        _, predicted = output.max(1)
        return predicted.item()

if len(test_dataset) > 0:
    sample_features, true_label = test_dataset[0]
    predicted = predict_removed_player(model, {k: v.unsqueeze(0) for k, v in sample_features.items()})
    print(f"Sample prediction: Player {predicted} (True: Player {true_label.item()})")