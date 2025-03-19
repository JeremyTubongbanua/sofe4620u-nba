# sofe4620u-nba

SOFE 4620U - NBA Lineup Prediction

## Authors

Group 6

- [Emily Lai](https://github.com/emilyirenelai) - 100825007
- [Jeremy Mark Tubongbanua](https://github.com/JeremyTubongbanua) - 100849092
- [Natasha Naorem](https://github.com/natt-n) - 100845321

## Project Objectives

Predict the 5th player to optimize the home team's chances of winning the game.

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
├── NBA 5th Player Prediction Report.pdf
├── NBA 5th Player Prediction Slides.pdf
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
- `NBA 5th Player Prediction Report.pdf`: The final report of the project.
- `NBA 5th Player Prediction Slides.pdf`: The final presentation slides of the project.
- `basketball_player_removal_mlp_model_3.pt`: The trained model for predicting the best basketball player to remove from a team. This is our 3rd and final iteration.
- `create_data.py`: A Python script that preprocesses the matchups data and NBA_test.csv and generates the `.txt`, `mapped_data.csv`, and `mapped_expanded.csv` files.
- `MLP_Training.ipynb`: A Jupyter Notebook that trains the MLP model for predicting the best basketball player to remove from a team.
- `NBA_testing.ipynb`: A Jupyter Notebook that tests the MLP model on the NBA_test.csv data and outputs the results.
- `nba_prediction_results.csv`: The final results of the predictions made on the NBA test data.

## Instructions for Setting up and Running The Code

You do not need to run `create_data.py`, since it was already created and saved in the `data.zip` for you. If you ran the file yourself, it would simply overwrite the existing files, which is not necessary.

If you would like to see the create_data.py script, you can run it for yourself. Otherwise, here's an output sample:

```bash
➜  sofe4620u-nba git:(trunk) ✗ python3 create_data.py 
Data directory already exists, skipping download and extraction.
Total rows in matchups-20*.csv: 236912 | First 5 rows: 
           game  season home_team away_team  starting_min        home_0  ...          away_0          away_1           away_2             away_3         away_4 outcome
0  201410290SAC    2015       SAC       GSW             0  Ben McLemore  ...    Andrew Bogut  Draymond Green  Harrison Barnes      Klay Thompson  Stephen Curry      -1
1  201410290SAC    2015       SAC       GSW             7  Ben McLemore  ...  Draymond Green    Festus Ezeli  Harrison Barnes      Klay Thompson  Stephen Curry      -1
2  201410290SAC    2015       SAC       GSW             8   Carl Landry  ...  Andre Iguodala    Festus Ezeli    Klay Thompson  Marreese Speights  Stephen Curry      -1
3  201410290SAC    2015       SAC       GSW             9   Carl Landry  ...  Andre Iguodala    Festus Ezeli    Klay Thompson  Marreese Speights  Stephen Curry      -1
4  201410290SAC    2015       SAC       GSW            10   Carl Landry  ...  Andre Iguodala    Festus Ezeli    Klay Thompson  Marreese Speights  Stephen Curry      -1

[5 rows x 16 columns]
Total unique names in matchups-20*.csv: 1044
Total unique names now with NBA_test.csv: 1084
Total unique names: 1084 | First 5 names: ['A.J. Price', 'Aaron Brooks', 'Aaron Gordon', 'Aaron Gray', 'Aaron McKie']
Total unique games: 10828 | First 5 games: ['200610310LAL', '200610310MIA', '200611010BOS', '200611010CHA', '200611010CLE']
Total unique teams in matchups-20*.csv: 35
Total unique teams now with NBA_test.csv: 35
Total unique teams: 35 | First 5 teams: ['ATL', 'BOS', 'BRK', 'CHA', 'CHI']
Total unique seasons in matchups-20*.csv: 9
Total unique seasons now with NBA_test.csv: 10
Total unique seasons: 10 | First 5 seasons: ['2007', '2008', '2009', '2010', '2011']
Created mapped_data.csv
   game  season  home_team  away_team  starting_min  outcome               home_players               away_players
0  9610       8         29         10             0       -1   [88, 241, 256, 476, 897]   [51, 307, 396, 617, 958]
1  9610       8         29         10             7       -1   [88, 241, 256, 476, 897]  [307, 355, 396, 617, 958]
2  9610       8         29         10             8       -1  [138, 241, 256, 785, 897]   [43, 355, 617, 713, 958]
3  9610       8         29         10             9       -1  [138, 241, 256, 277, 785]   [43, 355, 617, 713, 958]
4  9610       8         29         10            10       -1  [138, 277, 785, 843, 859]   [43, 355, 617, 713, 958]
Processed row 0/236912
Processed row 100000/236912
Processed row 200000/236912
Created mapped_expanded.csv (1184560 rows)
   game  season  home_team  away_team  starting_min  outcome          home_players              away_players  player_removed
0  9610       8         29         10             0       -1  [241, 256, 476, 897]  [51, 307, 396, 617, 958]              88
1  9610       8         29         10             0       -1   [88, 256, 476, 897]  [51, 307, 396, 617, 958]             241
2  9610       8         29         10             0       -1   [88, 241, 476, 897]  [51, 307, 396, 617, 958]             256
3  9610       8         29         10             0       -1   [88, 241, 256, 897]  [51, 307, 396, 617, 958]             476
4  9610       8         29         10             0       -1   [88, 241, 256, 476]  [51, 307, 396, 617, 958]             897
```

## Overview of Results

Running NBA_Testing.ipynb will output the following results. It will also output nba_test_results.csv, which contains the results of the predictions.

Using our 3rd iteration of the model, we achieved the following results:

28.90% overall accuracy → The model successfully guessed the right 5th player 28.90% of the time.
The 5th player was in the Top 10 candidates 73.00% of the time
The 5th player was in the Top 5 candidates 65.30% of the time
The 5th player was in the Top 3 candidates 53.90% of the time

See NBA_Testing.ipynb. This jupyter notebook created a final CSV of all predictions.

```sh
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

### Why We Think Our Prediction Accuracy Is Low

Some data observations:

- There are 40 new names in testing data in NBA_test.csv. The model has no prior data to these 40 new names, so it cannot make guesses on if it is good with other team members or if it should be a 5th player.
- There is a 10th season: NBA_test.csv has a 10th season: Season 2016. There is no past data on 2016, so predictions on 2016 games are unpredictable on who the model will recommend as the 5th player.
- 2 names that are not in the training data or the test data. NBA_test_labels.csv introduces two new names: “Anthony Brown” and “R.J. Hunter”, which are not found anywhere in `NBA_test.csv` and `matchups-20*.csv`. It is impossible for the model to ever get this prediction right, because it is a completely new name, as if they weren’t even NBA players.

Here's the list of 40 names not found in the training data:

```sh
names: ['Alan Williams', 'Larry Nance', 'Bobby Portis', 'Nikola Jokic', 'Christian Wood', 'Norman Powell', 'Tyus Jones', 'Frank Kaminsky', 'Devin Booker', 'Cameron Payne', 'Kristaps Porzingis', 'Josh Richardson', 'Emmanuel Mudiay', 'Nemanja Bjelica', 'Stanley Johnson', 'Jahlil Okafor', "D'Angelo Russell", 'Willie Cauley-Stein', 'Karl-Anthony Towns', 'Justise Winslow', 'T.J. McConnell', 'Myles Turner', 'Jordan McRae', 'Terry Rozier', 'Marcelo Huertas', 'Jonathon Simmons', 'Rondae Hollis-Jefferson', 'Mario Hezonja', 'Montrezl Harrell', 'Willie Reed', 'Lamar Patterson', 'Xavier Munford', 'Kelly Oubre', 'Raul Neto', 'Richaun Holmes', 'Trey Lyles', 'Chris McCullough', 'Rashad Vaughn', 'Delon Wright', 'Jerian Grant']
Total unique names: 1084 | First 5 names: ['A.J. Price', 'Aaron Brooks', 'Aaron Gordon', 'Aaron Gray', 'Aaron McKie']
```

After running `NBA_Testing.ipynb`, the following results we obtained `nba_predictions_results.csv`:

- On row 941: Our model was 44.44% confident that “Jordan Farmer” was the 5th optimal player, but the true answer was Anthony Brown. Since Anthony Brown’s name is not an output neuron at all (since his name does not exist in any of the training data), this was a failed attempt from the start.
- On row 910: Our model was 52.69% confident that “Luke Ridnour” was the 5th optimal player, but the true answer was Shabazz Muhammad. It was completely wrong because the Away Team included Alan Williams which was on of the 40 names that was not included in the Training Data.
- Training data only contained seasons 2007-2015. Testing data contained season 2016. The true player ranking standard deviation is substantially higher in 2016 than in the other seasons. This is expected, since the data our model was trained on is only for season 2007-2015. See Figure Below.

```bash
Standard Deviation of True Removed Player Rank by Season:
season
2007     43.070724
2008     30.179262
2009      9.848222
2010     28.969570
2011     38.944367
2012     83.959015
2013     47.577317
2014     18.266874
2015     32.809304
2016    216.385154
Name: true_player_rank, dtype: float64
```
