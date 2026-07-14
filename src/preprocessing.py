## we will work on preprocessing and clean the dataset # Scaling & preprocessing
 
"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""
 
# Import required libraries
import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Allow `from src...` imports when this file is executed as a script
# (e.g., `python src/preprocessing.py` from the project root or similar)
if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

 
 
class Preprocessor:
    """
    Preprocess the time series dataset.
    """
 
    def __init__(self):
        """
        Initialize the scaler.
        """
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
 
    def scale_data(self, df):
        """
        Scale the Passengers column.
 
        Parameters
        ----------
        df : pandas.DataFrame
 
        Returns
        -------
        scaled_df : pandas.DataFrame
        """
 
        print("\nOriginal Data")
        print(df.head())
 
        # Support either column naming: the loader uses `passengers`
        col_name = "Passengers" if "Passengers" in df.columns else "passengers"

        # Scale the passengers column
        scaled_values = self.scaler.fit_transform(df[[col_name]])

        # Return a consistent column name
        scaled_col = "Passengers"

 
        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=[scaled_col],
            index=df.index
        )

 
        # print("\nScaled Data")
        # print(scaled_df.head())
 
        # Save the scaler
        joblib.dump(self.scaler, "models/scaler.pkl")
 
        print("\nScaler saved successfully.")
 
        return scaled_df
 
if __name__ == "__main__":
 
    from src.data_loader import DataLoader
 
    # Use repo-relative path so it works across machines
    from pathlib import Path
    DATA_PATH = str(Path(__file__).resolve().parent.parent / "Data" / "airline-passengers.csv")

 
    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale data
    preprocessor = Preprocessor()
 
    scaled_df = preprocessor.scale_data(df)
 
    print("\nScaled Dataset")
    print(scaled_df.head())
 