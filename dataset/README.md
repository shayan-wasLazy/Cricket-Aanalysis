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
