import pandas as pd
from src.data.preprocess import load_and_clean_data


def test_missing_values_are_handled(tmp_path):
    # 1. Create a tiny fake dataset with a missing PM2.5 value (None)
    fake_data = pd.DataFrame(
        {
            "No": [1, 2],
            "year": [2010, 2010],
            "month": [1, 1],
            "day": [1, 1],
            "hour": [1, 2],
            "pm2.5": [15.0, None],  # ⚠️ Missing value here!
            "dew_point": [0, 0],
            "temp": [0, 0],
            "pressure": [1000, 1000],
            "wind_dir": ["NW", "NW"],
            "wind_speed": [1, 2],
            "snow": [0, 0],
            "rain": [0, 0],
        }
    )

    # 2. Save it to a temporary CSV file
    dummy_file = tmp_path / "dummy.csv"
    fake_data.to_csv(dummy_file, index=False)

    # 3. Run our actual preprocessing function on the fake file
    cleaned_df = load_and_clean_data(dummy_file)

    # 4. ASSERT (Check) that the missing value was successfully filled
    assert cleaned_df["pm2.5"].isnull().sum() == 0
