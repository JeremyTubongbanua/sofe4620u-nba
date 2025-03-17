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
print(f'Total rows in matchups-20*.csv: {len(dataframes)} | First 5 rows: \n{dataframes.head()}')

nba_test_df = pd.read_csv('data/NBA_test.csv')

# ---- Create unique lists of each column ----

# names
names = set()
for col in home_cols + away_cols:
    for row in dataframes[col]:
        names.add(row.strip())
print(f'Total unique names in matchups-20*.csv: {len(names)}')

for col in home_cols + away_cols:
    for row in nba_test_df[col]:
        names.add(row.strip())
        
names.remove('?')
        
print(f'Total unique names now with NBA_test.csv: {len(names)}')

names = sorted(list(names))

with open("data/names.txt", "w") as f:
    for name in names:
        f.write(name + "\n")

print(f"Total unique names: {len(names)} | First 5 names: {names[:5]}")

# games
games = []
for row in dataframes['game']:
    game = row.strip()
    if game not in games:
        games.append(game)
games.sort()

with open("data/games.txt", "w") as f:
    for game in games:
        f.write(game + "\n")
 
print(f"Total unique games: {len(games)} | First 5 games: {games[:5]}")

# teams
teams = set()
for row in dataframes['home_team']:
    teams.add(row.strip())
for row in dataframes['away_team']:
    teams.add(row.strip())
print(f'Total unique teams in matchups-20*.csv: {len(teams)}')

for row in nba_test_df['home_team']:
    teams.add(row.strip())
for row in nba_test_df['away_team']:
    teams.add(row.strip())

print(f'Total unique teams now with NBA_test.csv: {len(teams)}')

teams = sorted(list(teams))

with open("data/teams.txt", "w") as f:
    for team in teams:
        f.write(team + "\n")

print(f"Total unique teams: {len(teams)} | First 5 teams: {teams[:5]}")

# seasons
seasons = set()
for row in dataframes['season']:
    seasons.add(str(row))
print(f'Total unique seasons in matchups-20*.csv: {len(seasons)}')

for row in nba_test_df['season']:
    seasons.add(str(row))
print(f'Total unique seasons now with NBA_test.csv: {len(seasons)}')

seasons = sorted(list(seasons))

with open("data/seasons.txt", "w") as f:
    for season in seasons:
        f.write(season + "\n")

print(f"Total unique seasons: {len(seasons)} | First 5 seasons: {seasons[:5]}")

# ---- Create mapping dictionaries ----

names_dict = {name: i for i, name in enumerate(names)}
games_dict = {game: i for i, game in enumerate(games)}
teams_dict = {team: i for i, team in enumerate(teams)}
seasons_dict = {season: i for i, season in enumerate(seasons)}

# ---- Apply mappings to specific columns ----

# map names to indices
for col in home_cols + away_cols:
    dataframes[col] = dataframes[col].map(names_dict)

dataframes['game'] = dataframes['game'].map(games_dict) # map game id string to idx
dataframes['home_team'] = dataframes['home_team'].map(teams_dict) # map team id string to idx
dataframes['away_team'] = dataframes['away_team'].map(teams_dict) # map team id string to idx
dataframes['season'] = dataframes['season'].astype(str).map(seasons_dict) # map season int to idx

# ---- Create `home_players` and `away_players` column ----
# create home_players = [home_0, home_1, home_2, home_3, home_4]
# create away_players = [away_0, away_1, away_2, away_3, away_4]
# delete home_0, home_1, home_2, home_3, home_4, away_0, away_1, away_2, away_3, away_4
# save to 'data/mapped_data.csv'

dataframes['home_players'] = dataframes[home_cols].values.tolist()
dataframes['away_players'] = dataframes[away_cols].values.tolist()
dataframes['home_players'] = dataframes['home_players'].apply(sorted)
dataframes['away_players'] = dataframes['away_players'].apply(sorted)
dataframes.drop(columns=home_cols + away_cols, inplace=True)

# ---- Save mapped data ----
dataframes.to_csv('data/mapped_data.csv', index=False)
print('Created mapped_data.csv')
print(f'{dataframes.head()}')

# ----- Created mapped_expanded -----

df_copy = dataframes.copy()

for i in range(len(df_copy)):
    away_players = df_copy.iloc[i]['away_players']
    home_players = df_copy.iloc[i]['home_players']    
    random_index = random.randint(0, 4)
    player_removed = home_players.pop(random_index)
    df_copy.at[i, 'home_players'] = home_players
    df_copy.at[i, 'player_removed'] = int(player_removed)

df_copy['player_removed'] = df_copy['player_removed'].astype(int)

df_copy.to_csv("data/mapped_expanded.csv", index=False)
print('Created mapped_expanded.csv')
print(df_copy.head())
