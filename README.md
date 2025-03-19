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
├── basketball_player_removal_mlp_model_3.pt
├── create_data.py
├── MLP_Training.py
├── NBA_Testing.py
└── nba_prediction_results.csv
```

- `data/`: Contains all the data files used in the project.
  - `games.txt`: contains game strings
  - `names.txt`: contains all possible basketball player names
  - `seasons.txt`: contains season names
  - `teams.txt`: contains team names
- `data.zip`: A compressed file of the `data/` directory. Primarily used for data fetching from the cloud.
- `basketball_player_removal_mlp_model_3.pt`: The trained model for predicting the best basketball player to remove from a team. This is our 3rd and final iteration.
- `create_data.py`: A Python script that preprocesses the matchups data and NBA_test.csv and generates the `.txt`, `mapped_data.csv`, and `mapped_expanded.csv` files.
- `MLP_Training.ipynb`: A Jupyter Notebook that trains the MLP model for predicting the best basketball player to remove from a team.
- `NBA_testing.ipynb`: A Jupyter Notebook that tests the MLP model on the NBA_test.csv data and outputs the results.
- `nba_prediction_results.csv`: The final results of the predictions made on the NBA test data.

## Instructions for Setting up and Running The Code

You do not need to run `create_data.py`, since it was already created and saved in the `data.zip` for you. If you ran the file yourself, it would simply overwrite the existing files, which is not necessary.

If you would like to see the create_data.py script, you can run it for yourself. Otherwise, here's an output sample:

## Overview of Results

Running NBA_Testing.ipynb will output the following results. It will also output nba_test_results.csv, which contains the results of the predictions.

```
Running predictions on 1000 test samples...
100%|██████████| 1000/1000 [00:02<00:00, 443.06it/s]Top-1 accuracy: 28.90%
Top-3 accuracy: 53.90%
Top-5 accuracy: 65.30%
Top-10 accuracy: 77.00%
Overall accuracy: 28.90%
Mean rank of true player: 28.15
Median rank of true player: 3

Sample predictions:
   season home_team away_team  starting_min  \
0    2007       IND       BOS            18   
1    2007       HOU       DAL            16   
2    2007       SAS       POR            39   
3    2007       MIN       BOS            21   
4    2007       MEM       LAL            19   
5    2007       MIL       CLE             8   
6    2007       MIA       GSW            11   
7    2007       CLE       SAC            37   
8    2007       GSW       WAS            35   
9    2007       DEN       TOR            10   

                                        home_players  \
0  [Danny Granger, Darrell Armstrong, Keith McLeo...   
1  [Bonzi Wells, Juwan Howard, Luther Head, Tracy...   
2  [Beno Udrih, Bruce Bowen, Matt Bonner, Tim Dun...   
3  [Kevin Garnett, Randy Foye, Ricky Davis, Trent...   
4  [Chucky Atkins, Hakim Warrick, Mike Miller, Ru...   
5  [Andrew Bogut, Brian Skinner, Michael Redd, Mo...   
6  [Antoine Walker, James Posey, Jason Kapono, Ud...   
7  [Eric Snow, Larry Hughes, Sasha Pavlovic, Zydr...   
8  [Al Harrington, Andris Biedrins, Baron Davis, ...   
9  [Andre Miller, Carmelo Anthony, Joe Smith, Reg...   

                                        away_players  predicted_player  \
0  [Allan Ray, Gerald Green, Kendrick Perkins, Ry...   Jermaine O'Neal   
1  [Austin Croshere, Erick Dampier, Greg Buckner,...       Chuck Hayes   
2  [Brandon Roy, Jamaal Magloire, Jarrett Jack, J...     Manu Ginobili   
3  [Al Jefferson, Brian Scalabrine, Delonte West,...     Kevin Garnett   
4  [Andrew Bynum, Kobe Bryant, Lamar Odom, Luke W...     Darko Milicic   
5  [Anderson Varejao, Drew Gooden, Eric Snow, Lar...      Charlie Bell   
6  [Andris Biedrins, Jason Richardson, Matt Barne...       Dwyane Wade   
7  [Corliss Williamson, Francisco Garcia, John Sa...  Anderson Varejao   
8  [Antonio Daniels, Darius Songaila, Jarvis Haye...   Stephen Jackson   
9  [Chris Bosh, Joey Graham, Jorge Garbajosa, Jos...      Nene Hilario   

   confidence       true_player  true_player_rank  true_player_probability  \
0   36.400187       Troy Murphy               3.0                14.056769   
1   25.059590       Chuck Hayes               1.0                25.059589   
2   22.772816       Brent Barry               2.0                18.929443   
3   35.513160       Craig Smith               3.0                22.162634   
4   96.052271    Stromile Swift              14.0                 0.002492   
5   27.682203      Charlie Bell               1.0                27.682203   
6   26.970288       Dwyane Wade               1.0                26.970287   
7   72.234738  Anderson Varejao               1.0                72.234741   
8   31.178978   Mickael Pietrus               3.0                12.622147   
9   10.949451  Yakhouba Diawara              15.0                 2.318324   
```
