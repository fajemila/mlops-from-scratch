import pandas as pd
import os


def load_and_clean_data(input_path: str):
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("date").drop(columns=["No", "year", "month", "day", "hour"])
    df.columns = [
        "pm2.5",
        "dew_point",
        "temp",
        "pressure",
        "wind_dir",
        "wind_speed",
        "snow",
        "rain",
    ]

    df["pm2.5"] = df["pm2.5"].ffill()
    df = df.dropna()
    df = pd.get_dummies(df, columns=["wind_dir"], drop_first=True)
    return df


def split_and_save(df: pd.DataFrame, output_dir: str):
    split_index = int(len(df) * 0.8)
    train, test = df.iloc[:split_index], df.iloc[split_index:]

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"))
    test.to_csv(os.path.join(output_dir, "test.csv"))
    print(f"✅ Data processed and saved to {output_dir}")


if __name__ == "__main__":
    # When we run this script, it executes these steps:
    clean_df = load_and_clean_data("data/pollution.csv")
    split_and_save(clean_df, "data/processed")
