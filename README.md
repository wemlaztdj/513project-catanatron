# CS 513 Real-World Classification Project

This project uses seeded Catanatron simulations to answer two questions. Can we predict the winning seat immediately before the first dice roll? Do any simple measures of opening quality hold up across two bot policies and two table sizes?

Start with the [experiment report](results/REPORT.md). It contains all model accuracy scores and four confusion-matrix figures. The [classification reports](results/classification_report.csv) give precision, recall, F1, and support for every winning seat.

## Reproduce the experiment

The included results used Python 3.15.2 and the exact dependencies in `requirements.txt`. Run these commands from this directory in Windows PowerShell. Installing the pinned Catanatron revision requires Git.

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONHASHSEED = '0'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe simulate.py --runs 1000 --seed 513000 --workers 8
.\.venv\Scripts\python.exe evaluate.py
```

On macOS or Linux, create the environment with `python3.15 -m venv .venv`, use `.venv/bin/python` as the interpreter, and export the two environment variables with `export`.

An existing compatible environment can run the two scripts directly. Use `--workers 1` if subprocesses are unavailable. The simulation writes a partial CSV and replaces the final CSV after every game has completed. Evaluation reads that CSV and regenerates all other results.

## Data and experiment

Each scenario attempts seeds 513000-513999. Within a table size, the corresponding Basic and Advanced games share the board, seat order, and opening decisions. Basic uses Weighted Random after setup. Advanced uses Value Function with 5% exploration. All players at a table use the same post-setup bot policy. Strength labels describe the policies in this experiment. There are no human players or mixed-policy tables.

The simulator snapshots each player's positions at the first roll prompt and then completes the game to obtain the label. Each classifier is fit separately for each scenario to predict the winning seat. The feature vector contains the same opening measures for each seat. The shared seed split holds out 25% of unique seeds across all scenarios. See the [Catanatron engine](https://github.com/bcollazo/catanatron/tree/ecf931181b9a65bb4116a2153fb78c16f1438e00).

| Opening measure | Definition |
|---|---|
| Total, first, and second pips | Dice combinations for the resource tiles touching both settlements, the first settlement, or the second settlement. A shared tile counts for each settlement that can produce. |
| Resource-specific pips | Pips for each of wood, brick, sheep, wheat, and ore, measured before robber or trade effects. |
| Resource diversity and unique roll numbers | Number of resource types and distinct productive dice totals touching either settlement. |
| Starting cards and road cards | Number of resource cards granted by the second settlement and the number of those that are wood or brick. |
| Generic port and port fit | Access to a 3:1 port and the highest pip total among resources with a matching 2:1 port. |
| Expansion sites and best expansion pips | Count of legal settlement sites reachable from the opening road ends with one additional road and the largest pip total among these sites. |
| Road and city material pips | Minimum of wood and brick pips and minimum of wheat and ore pips, respectively. |
| Balanced score | Total pips + 2 x resource diversity + road material pips + city material pips. |

`openings.csv` records opening node and road IDs and the final number of turns for auditing. The classifiers use the opening measures in the table. Winner seat is the target. Seeds define the train/test partition. Feature and rule definitions are in `simulate.py`.

## Results files

| File | Contents |
|---|---|
| `REPORT.md` | Research findings, methodology, comparison tables, and figures. |
| `openings.csv` | Opening features and outcome for each player in each simulation. |
| `game_counts.csv` and `split.csv` | Scenario counts and seed assignments. |
| `model_metrics.csv` | Accuracy, macro precision, macro recall, macro F1, and baseline comparisons. |
| `classification_report.csv` | Per-seat, macro, and weighted precision, recall, F1, and support. |
| `confusion_*.png` and `confusion_counts.csv` | Labeled heatmaps and exact counts for each classifier and baseline. |
| `test_predictions.csv` | Held-out actual versus predicted winner for each model and game. |
| `opening_rule_results.csv` | Performance of seven opening rules and adjusted comparisons. |
| `cross_tier_results.csv` | Random Forest trained on one bot policy and tested on the other. |


