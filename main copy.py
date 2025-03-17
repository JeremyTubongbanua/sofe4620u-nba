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

df_copy = df.copy()

for i in range(len(df_copy)):
    away_team_list_of_players_str = df_copy.iloc[i]['away_players']
    away_team_list_of_players = list(map(int, away_team_list_of_players_str[1:-1].split(',')))
    df_copy.at[i, 'away_players'] = away_team_list_of_players
    home_players_list_of_players_str = df_copy.iloc[i]['home_players']
    home_players_list_of_players = list(map(int, home_players_list_of_players_str[1:-1].split(',')))
    df_copy.at[i, 'home_players'] = home_players_list_of_players

print(df_copy.head())

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
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
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

        outcome = self.predictor(torch.cat([team_encoding, team_diff], dim=1))

        return outcome


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

        outcome = torch.tensor([row['outcome']], dtype=torch.float)

        return home_data, away_data, context, outcome


def collate_fn(batch):
    home_data_list, away_data_list, contexts, outcomes = zip(*batch)

    home_batch = Batch.from_data_list(home_data_list)
    away_batch = Batch.from_data_list(away_data_list)

    contexts = torch.stack(contexts)
    outcomes = torch.stack(outcomes)

    return home_batch, away_batch, contexts, outcomes


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for home_data, away_data, context, target in train_loader:
            home_data = home_data.to(device)
            away_data = away_data.to(device)
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(home_data, away_data, context)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for home_data, away_data, context, target in val_loader:
                home_data = home_data.to(device)
                away_data = away_data.to(device)
                context = context.to(device)
                target = target.to(device)
                
                output = model(home_data, away_data, context)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses


def recommend_player(model, home_team_4players, available_players, away_team, context):
    """
    Recommend the best player to add to a 4-player home team
    
    Args:
        model: Trained GNN model
        home_team_4players: List of 4 player IDs
        available_players: List of available player IDs to choose from
        away_team: List of 5 away player IDs
        context: [season, starting_min]
    
    Returns:
        best_player: ID of the recommended player
        scores: Dict of {player_id: predicted_outcome}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    scores = {}

    for player_id in available_players:
        if player_id in home_team_4players:
            continue

        home_team_5players = home_team_4players + [player_id]

        home_players = torch.tensor(home_team_5players, dtype=torch.long).to(device)
        away_players = torch.tensor(away_team, dtype=torch.long).to(device)
        
        home_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    home_edges.append([i, j])
        home_edge_index = torch.tensor(home_edges, dtype=torch.long).t().contiguous().to(device)
        
        away_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    away_edges.append([i, j])
        away_edge_index = torch.tensor(away_edges, dtype=torch.long).t().contiguous().to(device)
        
        home_data = Data(x=home_players, edge_index=home_edge_index, batch=torch.zeros(5, dtype=torch.long).to(device))
        away_data = Data(x=away_players, edge_index=away_edge_index, batch=torch.zeros(5, dtype=torch.long).to(device))
        
        context_tensor = torch.tensor([context], dtype=torch.float).to(device)
        
        with torch.no_grad():
            predicted_outcome = model(home_data, away_data, context_tensor).item()
            scores[player_id] = predicted_outcome
    
    best_player = max(scores.items(), key=lambda x: x[1])[0]
    
    return best_player, scores


def main():
    df = pd.read_csv("data/mapped_data.csv", usecols=columns)
    
    def convert_player_list(player_str):
        return list(map(int, player_str[1:-1].split(',')))
    
    df['home_players'] = df['home_players'].apply(convert_player_list)
    df['away_players'] = df['away_players'].apply(convert_player_list)
    
    all_players = set()
    for players in df['home_players']:
        all_players.update(players)
    for players in df['away_players']:
        all_players.update(players)
    num_players = max(all_players) + 1
    
    train_df, val_df = train_test_split(df, test_size=1-SPLIT_RATIO, random_state=42)
    
    train_dataset = BasketballDataset(train_df, num_players)
    val_dataset = BasketballDataset(val_df, num_players)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = BasketballTeamGNN(num_players, EMBEDDING_DIM, HIDDEN_DIM)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    
    torch.save(model.state_dict(), 'basketball_gnn_model.pt')
    
    sample = val_df.iloc[0]
    
    home_team_4players = sample['home_players'][:4]
    available_players = list(all_players)
    away_team = sample['away_players']
    context = [sample['season'], sample['starting_min']]
    
    best_player, scores = recommend_player(model, home_team_4players, available_players, away_team, context)
    
    print(f"Original 4 players: {home_team_4players}")
    print(f"Recommended player to add: {best_player}")
    print(f"Top 5 recommendations:")
    top_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for player, score in top_players:
        print(f"Player {player}: Predicted outcome = {score:.4f}")


def evaluate_model_on_test_data():
    # Load the testing data
    test_columns = [
        "game", "season", "home_team", "away_team", "starting_min", "outcome",
        "home_players", "away_players", "player_removed"
    ]
    test_df = pd.read_csv("data/testing_data.csv", usecols=test_columns)
    
    # Only consider cases where the original outcome was positive
    positive_df = test_df[test_df['outcome'] == 1].copy()
    
    # Convert string representations of lists to actual lists
    def convert_player_list(player_str):
        return list(map(int, player_str[1:-1].split(',')))
    
    positive_df['home_players'] = positive_df['home_players'].apply(convert_player_list)
    positive_df['away_players'] = positive_df['away_players'].apply(convert_player_list)
    positive_df['player_removed'] = positive_df['player_removed'].apply(int)
    
    # Get all unique players
    all_players = set()
    for players in positive_df['home_players']:
        all_players.update(players)
    for players in positive_df['away_players']:
        all_players.update(players)
    all_players.update(positive_df['player_removed'])
    num_players = max(all_players) + 1
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasketballTeamGNN(num_players, EMBEDDING_DIM, HIDDEN_DIM)
    
    try:
        model.load_state_dict(torch.load('basketball_gnn_model.pt'))
        print("Loaded pre-trained model successfully.")
    except:
        print("Could not load pre-trained model. Please train the model first.")
        return
    
    model = model.to(device)
    model.eval()
    
    # Prepare for evaluation
    correct_predictions = 0
    top_5_hits = 0
    top_10_hits = 0
    reciprocal_ranks = []
    
    print(f"Evaluating on {len(positive_df)} test cases with positive outcomes...")
    
    for idx, row in positive_df.iterrows():
        # Create the 4-player home team by removing the player
        home_players = row['home_players'].copy()
        removed_player = row['player_removed']
        
        # Skip if the removed player isn't in the home team
        if removed_player not in home_players:
            continue
            
        # Create 4-player team by removing the specified player
        home_players.remove(removed_player)
        
        if len(home_players) != 4:
            continue  # Skip if we don't have exactly 4 players
        
        # Get prediction context
        away_team = row['away_players']
        context = [row['season'], row['starting_min']]
        
        # Get recommendations
        available_players = list(all_players)
        best_player, scores = recommend_player(model, home_players, available_players, away_team, context)
        
        # Sort players by predicted score
        ranked_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Check if the removed player was the top recommendation
        if best_player == removed_player:
            correct_predictions += 1
        
        # Check if the removed player was in the top 5
        top_5_players = [p[0] for p in ranked_players[:5]]
        if removed_player in top_5_players:
            top_5_hits += 1
        
        # Check if the removed player was in the top 10
        top_10_players = [p[0] for p in ranked_players[:10]]
        if removed_player in top_10_players:
            top_10_hits += 1
        
        # Calculate mean reciprocal rank (MRR)
        player_rank = next((i + 1 for i, (p, _) in enumerate(ranked_players) if p == removed_player), len(ranked_players) + 1)
        reciprocal_rank = 1.0 / player_rank if player_rank <= len(ranked_players) else 0
        reciprocal_ranks.append(reciprocal_rank)
        
        # Print details for this test case
        if idx < 5 or best_player == removed_player:  # Print first 5 examples and successful predictions
            print(f"\nTest case {idx}:")
            print(f"Game: {row['game']}, Season: {row['season']}")
            print(f"4-player team: {home_players}")
            print(f"Removed player: {removed_player}")
            print(f"Top 5 recommendations: {top_5_players}")
            print(f"Rank of removed player: {player_rank}")
    
    # Calculate metrics
    total_evaluated = len(reciprocal_ranks)
    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
    top_5_accuracy = top_5_hits / total_evaluated if total_evaluated > 0 else 0
    top_10_accuracy = top_10_hits / total_evaluated if total_evaluated > 0 else 0
    mrr = sum(reciprocal_ranks) / total_evaluated if total_evaluated > 0 else 0
    
    print("\nEvaluation Metrics:")
    print(f"Total test cases evaluated: {total_evaluated}")
    print(f"Exact match accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated})")
    print(f"Top-5 accuracy: {top_5_accuracy:.4f} ({top_5_hits}/{total_evaluated})")
    print(f"Top-10 accuracy: {top_10_accuracy:.4f} ({top_10_hits}/{total_evaluated})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    # Plot rank distribution
    plt.figure(figsize=(10, 6))
    plt.hist([next((i + 1 for i, (p, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)) 
                   if p == row['player_removed']), len(scores) + 1) 
              for _, row in positive_df.iterrows() if row['player_removed'] in row['home_players']],
             bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=1, color='r', linestyle='--', label='Exact Match')
    plt.axvline(x=5, color='g', linestyle='--', label='Top-5 Threshold')
    plt.axvline(x=10, color='orange', linestyle='--', label='Top-10 Threshold')
    plt.xlabel('Rank of Removed Player')
    plt.ylabel('Frequency')
    plt.title('Distribution of Removed Player Ranks')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rank_distribution.png')
    
    return accuracy, top_5_accuracy, top_10_accuracy, mrr


def player_analysis(model_path='basketball_gnn_model.pt'):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First, we need to determine the number of players
    df = pd.read_csv("data/mapped_data.csv", usecols=["home_players", "away_players"])
    
    def convert_player_list(player_str):
        return list(map(int, player_str[1:-1].split(',')))
    
    df['home_players'] = df['home_players'].apply(convert_player_list)
    df['away_players'] = df['away_players'].apply(convert_player_list)
    
    all_players = set()
    for players in df['home_players']:
        all_players.update(players)
    for players in df['away_players']:
        all_players.update(players)
    num_players = max(all_players) + 1
    
    # Load player names if available
    try:
        with open("data/names.txt", "r") as f:
            player_names = [line.strip() for line in f.readlines()]
        use_names = True
    except:
        player_names = [f"Player {i}" for i in range(num_players)]
        use_names = False
    
    # Load the model
    model = BasketballTeamGNN(num_players, EMBEDDING_DIM, HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
    except:
        print("Could not load the model. Please train the model first.")
        return
    
    # Extract player embeddings
    player_embeddings = model.player_embedding.weight.detach().cpu().numpy()
    
    # Analyze player similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(player_embeddings)
    
    # Find the most similar players for each player
    top_players = 10
    most_similar_players = {}
    
    for player_id in range(num_players):
        if player_id not in all_players:
            continue  # Skip players that don't appear in the dataset
            
        # Get similarity scores for this player
        similarities = similarity_matrix[player_id]
        
        # Get indices of most similar players (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_players+1]
        
        # Store the player ID and similarity score
        most_similar_players[player_id] = [(idx, similarities[idx]) for idx in similar_indices]
    
    # Print the most similar players for some examples
    num_examples = min(5, len(most_similar_players))
    example_players = random.sample(list(most_similar_players.keys()), num_examples)
    
    print("\nPlayer Similarity Analysis:")
    for player_id in example_players:
        player_name = player_names[player_id] if player_id < len(player_names) else f"Player {player_id}"
        print(f"\nMost similar players to {player_name} (ID: {player_id}):")
        
        for similar_id, similarity in most_similar_players[player_id][:5]:
            similar_name = player_names[similar_id] if similar_id < len(player_names) else f"Player {similar_id}"
            print(f"  - {similar_name} (ID: {similar_id}): similarity {similarity:.4f}")
    
    # Optional: visualize player embeddings with dimensionality reduction
    try:
        from sklearn.decomposition import PCA
        
        # Reduce embeddings to 2D for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(player_embeddings)
        
        # Plot a subset of players for clarity
        plt.figure(figsize=(12, 10))
        
        # Sample players to display (to avoid overcrowding)
        max_display = 100
        display_players = random.sample(list(all_players), min(max_display, len(all_players)))
        
        for player_id in display_players:
            plt.scatter(reduced_embeddings[player_id, 0], reduced_embeddings[player_id, 1], alpha=0.7)
            if use_names and player_id < len(player_names):
                plt.annotate(player_names[player_id], (reduced_embeddings[player_id, 0], reduced_embeddings[player_id, 1]))
            else:
                plt.annotate(str(player_id), (reduced_embeddings[player_id, 0], reduced_embeddings[player_id, 1]))
        
        plt.title('Player Embedding Visualization (PCA)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig('player_embeddings.png')
    except:
        print("Could not create PCA visualization (sklearn may be missing)")

    
# Add these lines to the end of the main function
def expanded_main():
    # First run the original main function to train the model
    main()
    
    # Then evaluate on the test data
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST DATA")
    print("="*50)
    accuracy, top5_acc, top10_acc, mrr = evaluate_model_on_test_data()
    
    # Optional: Analyze player embeddings and similarities
    print("\n" + "="*50)
    print("ANALYZING PLAYER EMBEDDINGS")
    print("="*50)
    player_analysis()
    
    return accuracy, top5_acc, top10_acc, mrr


if __name__ == "__main__":
    expanded_main()