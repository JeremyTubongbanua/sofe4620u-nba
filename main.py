# ---- Hyperparameters ----
EPOCHS = 2
BATCH_SIZE=32
SPLIT_RATIO=0.8
LEARNING_RATE = 0.001

# ---- Imports ----
import requests
import io
import zipfile
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
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

# ---- Define relevant columns ----
relevant_columns = [
    'game', # e.g. '201010270DEN'
    'season', # e.g. '2011'
    'home_team', # e.g. 'DEN'
    'away_team', # e.g. 'UTA'
    'starting_min', # e.g. 0 or 3 or 14
    'home_0', # person's name
    'home_1', # person's name
    'home_2', # person's name
    'home_3', # person's name
    'home_4', # person's name
    'away_0', # person's name
    'away_1', # person's name
    'away_2', # person's name
    'away_3', # person's name
    'away_4', # person's name
    'outcome' # -1, 0, or 1
]

# ---- Load data ----

matchup_files = glob.glob('data/matchups-20*.csv')

dataframes = []
for file in matchup_files:
    df = pd.read_csv(file)
    df = df[relevant_columns]
    dataframes.append(df)

dataframes = pd.concat(dataframes, ignore_index=True)
print(f'Total rows: {len(dataframes)} | First 5 rows: \n{dataframes.head()}')

# ---- Preprocess data 1 - convert names from home_0, home_1, home_2, home_3, home_4 and away_0, away_1, away_2, away_3, away_4 to unique idx ----

names = []
names.extend(dataframes['home_0'].unique())
names.extend(dataframes['home_1'].unique())
names.extend(dataframes['home_2'].unique())
names.extend(dataframes['home_3'].unique())
names.extend(dataframes['home_4'].unique())
names.extend(dataframes['away_0'].unique())
names.extend(dataframes['away_1'].unique())
names.extend(dataframes['away_2'].unique())
names.extend(dataframes['away_3'].unique())
names.extend(dataframes['away_4'].unique())
names = list(set(names))
names.sort()
print(f"Total unique names: {len(names)} | First 10 names: {names[:10]}")

# ---- Preprocess data 2 - `game` (e.g. '201410290SAC') to unique idx ----

games = dataframes['game'].unique()
games = list(games)
games.sort()
print(f"Total unique games: {len(games)} | First 10 games: {games[:10]}")

# ---- Preprocess data 3 - `season` (e.g. '2011') to unique idx ----

seasons = dataframes['season'].unique()
seasons = list(seasons)
seasons.sort()
print(f"Total unique seasons: {len(seasons)} | Seasons: {str(seasons)}")

# ---- Preprocess data 4 - `home_team` and `away_team` (e.g. 'DEN') to unique idx ----

teams = []
teams.extend(dataframes['home_team'].unique())
teams.extend(dataframes['away_team'].unique())
teams = list(set(teams))
teams.sort()
print(f"Total unique teams: {len(teams)} | First 10 teams: {teams[:10]}")

# ---- Create a new dataframe that uses indices instead ----

dataframes['home_0'] = dataframes['home_0'].apply(lambda x: names.index(x))
dataframes['home_1'] = dataframes['home_1'].apply(lambda x: names.index(x))
dataframes['home_2'] = dataframes['home_2'].apply(lambda x: names.index(x))
dataframes['home_3'] = dataframes['home_3'].apply(lambda x: names.index(x))
dataframes['home_4'] = dataframes['home_4'].apply(lambda x: names.index(x))
dataframes['away_0'] = dataframes['away_0'].apply(lambda x: names.index(x))
dataframes['away_1'] = dataframes['away_1'].apply(lambda x: names.index(x))
dataframes['away_2'] = dataframes['away_2'].apply(lambda x: names.index(x))
dataframes['away_3'] = dataframes['away_3'].apply(lambda x: names.index(x))
dataframes['away_4'] = dataframes['away_4'].apply(lambda x: names.index(x))
dataframes['game'] = dataframes['game'].apply(lambda x: games.index(x))
dataframes['season'] = dataframes['season'].apply(lambda x: seasons.index(x))
dataframes['home_team'] = dataframes['home_team'].apply(lambda x: teams.index(x))
dataframes['away_team'] = dataframes['away_team'].apply(lambda x: teams.index(x))

# sort home_0, home_1, home_2, home_3, home_4 and away_0, away_1, away_2, away_3, away_4
dataframes['home_players'] = dataframes[['home_0', 'home_1', 'home_2', 'home_3', 'home_4']].values.tolist()
dataframes['home_players'] = dataframes['home_players'].apply(lambda x: sorted(x))
dataframes['away_players'] = dataframes[['away_0', 'away_1', 'away_2', 'away_3', 'away_4']].values.tolist()
dataframes['away_players'] = dataframes['away_players'].apply(lambda x: sorted(x))

print(f'Total rows: {len(dataframes)} | First 3 rows: \n{dataframes.head(3)}')

split_idx = int(len(dataframes) * SPLIT_RATIO)
train_df = dataframes[:split_idx]
test_df = dataframes[split_idx:]

print(f"Train data # rows: {len(train_df)} | Test data # rows: {len(test_df)}")

# ---- Create a custom dataset ----

class NBAData(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        home_players = [row['home_0'], row['home_1'], row['home_2'], row['home_3'], row['home_4']]
        random_player_to_remove = torch.randint(0, 5, (1,)).item()
        home_players.pop(random_player_to_remove)
        home_players.sort()
        home_players_idx = torch.tensor(home_players, dtype=torch.uint32)
        away_players = [row['away_0'], row['away_1'], row['away_2'], row['away_3'], row['away_4']]
        away_players.sort()
        away_players_idx = torch.tensor(away_players, dtype=torch.uint32)
        return {
            'game_idx': torch.tensor(row['game'], dtype=torch.uint32),
            'season_idx': torch.tensor(row['season'], dtype=torch.uint32),
            'home_team_idx': torch.tensor(row['home_team'], dtype=torch.uint32),
            'away_team_idx': torch.tensor(row['away_team'], dtype=torch.uint32),
            'starting_min': torch.tensor(row['starting_min'], dtype=torch.uint32),
            'home_players_idx': home_players_idx,
            'away_players_idx': away_players_idx,
            'outcome': torch.tensor(row['outcome'], dtype=torch.int8),
            'random_player_removed_idx': torch.tensor(row[f'home_{random_player_to_remove}'], dtype=torch.uint32)
        }
        
train_dataset = NBAData(train_df)
test_dataset = NBAData(test_df)

print (f"Train dataset # rows: {len(train_dataset)} | Test dataset # rows: {len(test_dataset)}")
sample_data = train_dataset[0]
print(f"Sample data: {sample_data}")

# ---- Create train and test dataloaders ----

