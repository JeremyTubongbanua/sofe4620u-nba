# ---- Hyperparameters ----
EPOCHS = 2
BATCH_SIZE = 32
SPLIT_RATIO = 0.8
LEARNING_RATE = 0.001

# ---- Imports ----
import os
import glob
import pandas as pd
import requests
import zipfile
import random

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
    'game',  # e.g. '201010270DEN'
    'season',  # e.g. '2011'
    'home_team',  # e.g. 'DEN'
    'away_team',  # e.g. 'UTA'
    'starting_min',  # e.g. 0 or 3 or 14
    'home_0',  # person's name
    'home_1',  # person's name
    'home_2',  # person's name
    'home_3',  # person's name
    'home_4',  # person's name
    'away_0',  # person's name
    'away_1',  # person's name
    'away_2',  # person's name
    'away_3',  # person's name
    'away_4',  # person's name
    'outcome'  # -1, 0, or 1
]

home_cols = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']
away_cols = ['away_0', 'away_1', 'away_2', 'away_3', 'away_4']

# ---- Load data ----
matchup_files = glob.glob('data/matchups-20*.csv')

dataframes = []
for file in matchup_files:
    df = pd.read_csv(file)
    df = df[relevant_columns]
    dataframes.append(df)

dataframes = pd.concat(dataframes, ignore_index=True)
print(f'Total rows: {len(dataframes)} | First 5 rows: \n{dataframes.head()}')

# ---- Create unique lists of each column ----

# names
if not os.path.exists("data/names.txt"):
    names = set()
    for col in home_cols + away_cols:
        for row in dataframes[col]:
            names.add(row.strip())
    names = sorted(list(names))

    with open("data/names.txt", "w") as f:
        for name in names:
            f.write(name + "\n")
else:
    with open("data/names.txt", "r") as f:
        names = f.read().splitlines()

print(f"Total unique names: {len(names)} | First 5 names: {names[:5]}")

# games
if not os.path.exists("data/games.txt"):
    games = []
    for row in dataframes['game']:
        game = row.strip()
        if game not in games:
            games.append(game)
    games.sort()

    with open("data/games.txt", "w") as f:
        for game in games:
            f.write(game + "\n")
else:
    with open("data/games.txt", "r") as f:
        games = f.read().splitlines()

print(f"Total unique games: {len(games)} | First 5 games: {games[:5]}")

# teams
if not os.path.exists("data/teams.txt"):
    teams = set()
    for row in dataframes['home_team']:
        teams.add(row.strip())
    for row in dataframes['away_team']:
        teams.add(row.strip())
    teams = sorted(list(teams))

    with open("data/teams.txt", "w") as f:
        for team in teams:
            f.write(team + "\n")
else:
    with open("data/teams.txt", "r") as f:
        teams = f.read().splitlines()

print(f"Total unique teams: {len(teams)} | First 5 teams: {teams[:5]}")

# seasons
if not os.path.exists("data/seasons.txt"):
    seasons = set()
    for row in dataframes['season']:
        seasons.add(str(row))
    seasons = sorted(list(seasons))

    with open("data/seasons.txt", "w") as f:
        for season in seasons:
            f.write(season + "\n")
else:
    with open("data/seasons.txt", "r") as f:
        seasons = f.read().splitlines()
        
print(f"Total unique seasons: {len(seasons)} | First 5 seasons: {seasons[:5]}")

# ---- Create mapping dictionaries ----

names_dict = {name: i for i, name in enumerate(names)}
games_dict = {game: i for i, game in enumerate(games)}
teams_dict = {team: i for i, team in enumerate(teams)}
seasons_dict = {season: i for i, season in enumerate(seasons)}

# ---- Apply mappings to specific columns ----

# print(f'{dataframes.head()}')

for col in home_cols + away_cols:
    dataframes[col] = dataframes[col].map(names_dict)

dataframes['game'] = dataframes['game'].map(games_dict)
dataframes['home_team'] = dataframes['home_team'].map(teams_dict)
dataframes['away_team'] = dataframes['away_team'].map(teams_dict)
dataframes['season'] = dataframes['season'].astype(str).map(seasons_dict)

# print(f'{dataframes.head()}')

# ---- Save mapped data ----
# create home_players = [home_0, home_1, home_2, home_3, home_4]
# create away_players = [away_0, away_1, away_2, away_3, away_4]
# delete home_0, home_1, home_2, home_3, home_4, away_0, away_1, away_2, away_3, away_4
# save to 'data/mapped_data.csv'

dataframes['home_players'] = dataframes[home_cols].values.tolist()
dataframes['away_players'] = dataframes[away_cols].values.tolist()
dataframes['home_players'] = dataframes['home_players'].apply(sorted)
dataframes['away_players'] = dataframes['away_players'].apply(sorted)
dataframes.drop(columns=home_cols + away_cols, inplace=True)


print(f'{dataframes.head()}')

dataframes.to_csv('data/mapped_data.csv', index=False)

# -----

with open("data/games.txt", "r") as f:
    games = {line.strip(): i for i, line in enumerate(f.readlines())}

with open("data/names.txt", "r") as f:
    names = [line.strip() for line in f.readlines()]

with open("data/seasons.txt", "r") as f:
    seasons = {line.strip(): i for i, line in enumerate(f.readlines())}

with open("data/teams.txt", "r") as f:
    teams = [line.strip() for line in f.readlines()]

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
    random_index = random.randint(0, 4)
    player_removed = home_players_list_of_players.pop(random_index)
    df_copy.at[i, 'home_players'] = home_players_list_of_players
    df_copy.at[i, 'player_removed'] = int(player_removed)

df_copy['player_removed'] = df_copy['player_removed'].astype(int)

training_dataframes = df_copy.sample(frac=SPLIT_RATIO, random_state=42)
testing_dataframes = df_copy.drop(training_dataframes.index)

print(f'Training dataframe head: \n{training_dataframes.head()}')

# Save to training and testing dataframes
training_dataframes.to_csv('data/training_data.csv', index=False)
testing_dataframes.to_csv('data/testing_data.csv', index=False)