# %%cell 1
import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

fastf1.Cache.enable_cache("./f1_data")  # Stores downloaded data locally


# %%cell 2
def build_dataset_for_race(year, round_num):
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=True)

    laps = session.laps
    drivers = laps["Driver"].unique()
    dataset = []

    for drv in tqdm(drivers):
        drv_laps = laps.pick_driver(drv).reset_index(drop=True)
        if len(drv_laps) < 3:
            continue  # not enough laps

        for i in range(len(drv_laps) - 1):
            current_lap = drv_laps.iloc[i]
            next_lap = drv_laps.iloc[i + 1]

            # Skip if lap is incomplete or telemetry is broken
            if pd.isna(current_lap["LapTime"]) or pd.isna(next_lap["LapTime"]):
                continue

            row = {
                "driver": drv,
                "lap_number": current_lap["LapNumber"],
                "compound": current_lap["Compound"],
                "tyre_age": current_lap["TyreLife"],
                "track_status": current_lap["TrackStatus"],  # 1: green, 4: SC
                "is_pit": current_lap["PitOutTime"] is not pd.NaT
                or current_lap["PitInTime"] is not pd.NaT,
                "position": current_lap["Position"],
                "lap_time": current_lap["LapTime"].total_seconds(),
                "will_pit_next_lap": int(next_lap["PitInTime"] is not pd.NaT),
            }

            dataset.append(row)

    return pd.DataFrame(dataset)


# %%cell 3
all_data = []

for year in [2021, 2022, 2023]:
    for round_num in range(1, 23):  # max rounds per season
        try:
            df = build_dataset_for_race(year, round_num)
            all_data.append(df)
        except Exception as e:
            print(f"Skipped {year} R{round_num}: {e}")

full_df = pd.concat(all_data).reset_index(drop=True)
full_df.to_csv("f1_pitstop_dataset.csv", index=False)

# %%cell 4
df = pd.read_csv("f1_pitstop_dataset.csv")

# Encode compound type
df["compound"] = df["compound"].astype("category").cat.codes

# Normalize lap_time and tyre_age
df["lap_time"] = df["lap_time"] / df["lap_time"].max()
df["tyre_age"] = df["tyre_age"] / df["tyre_age"].max()

# Drop non-numeric fields for now
X = df[["lap_number", "compound", "tyre_age", "track_status", "position", "lap_time"]]
y = df["will_pit_next_lap"]
