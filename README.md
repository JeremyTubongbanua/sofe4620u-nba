# sofe4620u-nba

SOFE 4620U - NBA Lineup Prediction

## Authors

Group 6

- [Emily Lai](https://github.com/emilyirenelai) - 100825007
- [Jeremy Mark Tubongbanua](https://github.com/JeremyTubongbanua) - 100849092
- [Natasha Naorem](https://github.com/natt-n) - 100845321

## Project Objectives

## Project Structure

Below is the project structure of the repository:

```md
.
├── data/
│   ├── games.txt
│   ├── names.txt
│   ├── seasons.txt
│   ├── teams.txt
│   ├── matchups-20*.tcsv
│   ├── NBA_test*.csv
│   ├── mapped_data.csv
│   └── mapped_expanded.csv
├── data.zip
├── basketball_player_removal_mlp_model.pt
├── create_data.py
└── MLP_Training.ipynb
```

- `data/`: Contains all the data files used in the project.
  - `games.txt`: contains game strings
  - `names.txt`: contains all possible basketball player names
  - `seasons.txt`: contains season names
  - `teams.txt`: contains team names
- `data.zip`: A compressed file of the `data/` directory. Primarily used for data fetching from the cloud.
- `basketball_player_removal_mlp_model.pt`: The trained model for predicting the best basketball player to remove from a team.
- `create_data.py`: A Python script that preprocesses the matchups data and NBA_test.csv and generates the `.txt`, `mapped_data.csv`, and `mapped_expanded.csv` files.
- `MLP_Training.ipynb`: A Jupyter Notebook that trains the MLP model for predicting the best basketball player to remove from a team.

## Instructions for Setting up and Running The Code

You do not need to run `create_data.py`, since it was already created and saved in the `data.zip` for you. If you ran the file yourself, it would simply overwrite the existing files, which is not necessary.

If you would like to see the create_data.py script, you can run it for yourself. Otherwise, here's an output sample:

## Overview of Results
