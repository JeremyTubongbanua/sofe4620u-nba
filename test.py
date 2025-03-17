# ---- Hyperparameters ----
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
device='cpu'

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


columns = [
    "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4"
]

df = pd.read_csv("data/NBA_test.csv", usecols=columns)

# make a blank dataframe
df_new = df.copy()

# convert team names to indices
df_new["home_team"] = df_new["home_team"].apply(lambda x: teams.index(x))
df_new["away_team"] = df_new["away_team"].apply(lambda x: teams.index(x))
df_new["season"] = df_new["season"].apply(lambda x: seasons[str(x)])

# print(df_new.head())   

# aggregate home_0 to home_4, if the value is '?', then skip it
home_players_list = []
away_players_list = []

for row in range(len(df_new)):
    home_players = []
    away_players = []
    
    for i in range(5):
        col_name = f"home_{i}"
        if df_new[col_name][row] != "?":
            name = df_new[col_name][row]
            if name not in names:
                names.append(name)  # Append new name to the names list
                with open("data/names.txt", "a") as f:  # Append to the names file
                    f.write(name + "\n")
            home_player_idx = names.index(name)
            home_players.append(home_player_idx)
                
    home_players_list.append(home_players)
    
    for i in range(5):
        col_name = f"away_{i}"
        if df_new[col_name][row] != "?":
            name = df_new[col_name][row]
            if name not in names:
                names.append(name)  # Append new name to the names list
                with open("data/names.txt", "a") as f:  # Append to the names file
                    f.write(name + "\n")
            away_player_idx = names.index(name)
            away_players.append(away_player_idx)
    
    away_players_list.append(away_players)

df_new['home_players'] = home_players_list
df_new['away_players'] = away_players_list
    
# remove home_0 to home_4, away_0 to away_4
df_new = df_new.drop(columns=["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"])

print(df_new.head())
    
# use the model in ./basketball_player_prediction_model.pt

# ---- Define the GNN Model ----
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

# Function to predict players to add to incomplete teams
def predict_players_to_add(model, home_players, away_players, season, starting_min, num_to_predict=1):
    """
    Predict players to add to an incomplete home team.
    
    Args:
        model: Trained BasketballTeamGNN model
        home_players: List of indices for home players (incomplete lineup)
        away_players: List of indices for away players
        season: Season index
        starting_min: Starting minute
        num_to_predict: Number of players to predict
        
    Returns:
        List of indices of players to add
    """
    model.eval()
    
    # Convert inputs to tensors
    home_players_tensor = torch.tensor(home_players, dtype=torch.long).to(device)
    away_players_tensor = torch.tensor(away_players, dtype=torch.long).to(device)
    
    # Create graph edges
    home_team_size = len(home_players)
    away_team_size = len(away_players)
    
    home_edges = []
    for i in range(home_team_size):
        for j in range(home_team_size):
            if i != j:
                home_edges.append([i, j])
    
    if not home_edges:  # If there's only one player, create a self-loop
        home_edges = [[0, 0]]
        
    home_edge_index = torch.tensor(home_edges, dtype=torch.long).t().contiguous().to(device)
    
    away_edges = []
    for i in range(away_team_size):
        for j in range(away_team_size):
            if i != j:
                away_edges.append([i, j])
                
    if not away_edges:  # If there's only one player, create a self-loop
        away_edges = [[0, 0]]
        
    away_edge_index = torch.tensor(away_edges, dtype=torch.long).t().contiguous().to(device)
    
    # Create data objects
    home_data = Data(
        x=home_players_tensor, 
        edge_index=home_edge_index, 
        batch=torch.zeros(home_team_size, dtype=torch.long).to(device)
    )
    
    away_data = Data(
        x=away_players_tensor, 
        edge_index=away_edge_index, 
        batch=torch.zeros(away_team_size, dtype=torch.long).to(device)
    )
    
    context = torch.tensor([[season, starting_min]], dtype=torch.float).to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(home_data, away_data, context)
        probabilities = F.softmax(logits, dim=1)
        
    # Get top predictions (excluding players already in the team)
    home_players_set = set(home_players)
    
    # Create a mask to zero out probabilities of players already in the team
    mask = torch.ones_like(probabilities)
    for player in home_players:
        mask[0, player] = 0
    
    masked_probs = probabilities * mask
    _, top_players = torch.topk(masked_probs, k=num_to_predict + 10)  # Get a few extra in case we need them
    
    # Convert to list and filter out any players that might be in the home team
    # (just a safeguard in case the masking didn't work perfectly)
    top_players_list = top_players.cpu().numpy()[0].tolist()
    filtered_players = [p for p in top_players_list if p not in home_players_set]
    
    return filtered_players[:num_to_predict]

# Load the pre-trained model
num_players = len(names)  # Total number of players in the dataset
print(f"Total number of players: {num_players}")

model = BasketballTeamGNN(num_players, EMBEDDING_DIM, HIDDEN_DIM)

# Check if the model file exists
if os.path.exists("basketball_player_prediction_model.pt"):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load("basketball_player_prediction_model.pt", map_location=device))
    model = model.to(device)
else:
    print("Pre-trained model not found. Please make sure 'basketball_player_prediction_model.pt' exists.")

# Analyze test data and make predictions
predictions = []
results_df = pd.DataFrame()

print("\nMaking predictions on test data...")
for idx, row in tqdm(df_new.iterrows(), total=len(df_new)):
    home_team = row['home_players']
    away_team = row['away_players']
    season = row['season']
    starting_min = row['starting_min']
    
    # Check if home team is incomplete (less than 5 players)
    players_to_add = 5 - len(home_team)
    if players_to_add > 0:
        predicted_players = predict_players_to_add(
            model, 
            home_team, 
            away_team, 
            season, 
            starting_min, 
            num_to_predict=players_to_add
        )
        
        # Store the prediction results
        prediction_entry = {
            'game_idx': idx,
            'home_team_incomplete': ', '.join([names[p] for p in home_team]),
            'away_team': ', '.join([names[p] for p in away_team]),
            'season': list(seasons.keys())[list(seasons.values()).index(season)],
            'starting_min': starting_min,
            'predicted_players': ', '.join([names[p] for p in predicted_players]),
            'predicted_player_ids': predicted_players
        }
        predictions.append(prediction_entry)

# Create results dataframe
if predictions:
    results_df = pd.DataFrame(predictions)
    print("\nPrediction Results:")
    print(results_df.head())
    
    # Save predictions to CSV
    results_df.to_csv('nba_player_predictions.csv', index=False)
    print("Predictions saved to 'nba_player_predictions.csv'")
else:
    print("No incomplete teams found in the test data.")

# Create a function to visualize player recommendations
def plot_top_recommended_players(results_df, top_n=10):
    if not len(results_df):
        print("No predictions to visualize")
        return
    
    # Count frequency of recommended players
    player_counts = {}
    for player_list in results_df['predicted_player_ids'].tolist():
        for player_id in player_list if isinstance(player_list, list) else [player_list]:
            player_name = names[player_id]
            player_counts[player_name] = player_counts.get(player_name, 0) + 1
    
    # Sort by frequency
    sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    player_names = [p[0] for p in sorted_players]
    counts = [p[1] for p in sorted_players]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(player_names)), counts, color='skyblue')
    plt.xticks(range(len(player_names)), player_names, rotation=45, ha='right')
    plt.xlabel('Player')
    plt.ylabel('Recommendation Count')
    plt.title(f'Top {top_n} Recommended Players')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            str(count),
            ha='center'
        )
    
    plt.tight_layout()
    plt.savefig('top_recommended_players.png')
    plt.show()

# Call the visualization function
if len(results_df) > 0:
    plot_top_recommended_players(results_df)

print("\nCode execution complete!")