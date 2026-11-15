# Competition dataset

Place the six SupeRR Selector source CSV files directly in this folder. The folder is tracked in Git, but the CSV files are intentionally ignored by `.gitignore` so the competition data is not uploaded when you push.

Required exact filenames:

```text
dataset/
├── ball_by_ball.csv
├── matches.csv
├── competitions.csv
├── teams.csv
├── playing_eleven.csv
└── competition_team_player_mapping.csv
```

The loader also accepts gzip-compressed files (for example `ball_by_ball.csv.gz`, `matches.csv.gz`).

## After cloning

1. Download the official/authorized SupeRR Selector data pack.
2. Copy the six files above into this folder (or use the corresponding `.csv.gz` files).
3. Do not rename columns or alter the source schema.
4. From the repository root, run:

   ```bash
   pip install -r requirements.txt
   python run_analysis.py
   ```

5. Open the notebook or launch the dashboard:

   ```bash
   jupyter lab notebooks/SupeRR_Pressure_Modelling_Comparison.ipynb
   streamlit run dashboard/app.py
   ```

## Verify Git is ignoring the data

From the repository root:

```bash
git check-ignore dataset/ball_by_ball.csv
git status --short
```

The first command should print the file path. Dataset files should not appear as untracked files in `git status`.

**Data handling:** these are competition-provided files. Keep them local unless the competition's terms explicitly allow redistribution. The project code and notebook can be pushed without pushing the data.


## Rajasthan Royals Squad Intelligence data (separate from the competition pack)

The RR analysis uses Cricsheet's IPL JSON archive rather than the six SupeRR Selector CSV files. The script downloads the archive to this ignored local path:

dataset/rr_intelligence/ipl_json.zip

Run from the repository root:

python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026

Alternatively, provide an already-downloaded ZIP or extracted JSON directory with --source PATH. Generated RR reports, delivery extracts and player-fit rankings are stored in outputs/rr_intelligence/ and remain ignored by Git.

Source: https://cricsheet.org/  
JSON format: https://cricsheet.org/format/json/  
Licence and attribution: https://cricsheet.org/licensing/

The source archive and competition-provided files are not committed. Preserve Cricsheet's ODC-By attribution and licence obligations if publishing derived data.


### Required model for PI / XGBoost phase ranking

The RR phase selector uses the same trained model saved by the main project analysis:

models/synthetic_pressure_xgb.json

Run the main analysis once before the RR build if the model file does not exist:

python run_analysis.py
python scripts/build_rr_intelligence.py --download --start-year 2022 --end-year 2026

Do not commit the model file or generated RR outputs; both are ignored by Git.
