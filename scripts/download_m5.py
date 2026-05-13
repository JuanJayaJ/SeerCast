from pathlib import Path

import kagglehub


RAW_DIR = Path("dataset")
RAW_DIR.mkdir(parents=True, exist_ok=True)

path = kagglehub.competition_download(
    "m5-forecasting-accuracy",
    output_dir=str(RAW_DIR),
    force_download=True,
)

print("Downloaded to:", path)

print("Files in dataset:")
for file in sorted(RAW_DIR.rglob("*")):
    if file.is_file():
        print(" -", file.relative_to(RAW_DIR))