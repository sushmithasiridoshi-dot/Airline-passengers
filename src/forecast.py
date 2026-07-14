# Future forecasting
 
"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""
 
import numpy as np
import joblib
 
from tensorflow.keras.models import load_model
 
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
 
 
class Forecaster:
 
    def __init__(self):
 
        self.data_path = "/Users/bouroju/Documents/teck-works/Deep-Learning/Air-line-passanger-forcasting/Data/airline-passengers.csv"
 
        self.model_path = "models/lstm_model.keras"
 
        self.scaler_path = "models/scaler.pkl"
 
        self.sequence_length = 12
 
    def forecast(self, future_months=12):
 
        # --------------------------
        # Load Dataset
        # --------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # --------------------------
        # Scale Dataset
        # --------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # --------------------------
        # Load Model
        # --------------------------
 
        model = load_model(self.model_path)
 
        # --------------------------
        # Load Scaler
        # --------------------------
 
        scaler = joblib.load(self.scaler_path)
 
        # --------------------------
        # Last 12 Months
        # --------------------------
 
        last_sequence = scaled_df.values[-self.sequence_length:]
 
        future_predictions = []
 
        # --------------------------
        # Forecast Loop
        # --------------------------
 
        for _ in range(future_months):
 
            input_data = last_sequence.reshape(
                1,
                self.sequence_length,
                1
            )
 
            prediction = model.predict(
                input_data,
                verbose=0
            )
 
            future_predictions.append(prediction[0,0])
 
            last_sequence = np.vstack(
                (
                    last_sequence[1:],
                    prediction
                )
            )
 
        # --------------------------
        # Convert Back
        # --------------------------
 
        future_predictions = np.array(
            future_predictions
        ).reshape(-1,1)
 
        future_predictions = scaler.inverse_transform(
            future_predictions
        )
 
        print("\nFuture Forecast Completed.")
 
        return future_predictions
if __name__ == "__main__":
 
    forecaster = Forecaster()
 
    future = forecaster.forecast(future_months=12)
 
    print("\nNext 12 Month Forecast\n")
 
    for i, value in enumerate(future, start=1):
 
        print(f"Month {i} : {value[0]:.2f}")
   
 
# Graph plotting
 
"""
====================================================
Module : visualization.py
Project: Airline Passenger Forecasting
Purpose: Visualize Model Performance
====================================================
"""
 
import matplotlib.pyplot as plt
 
 
class Visualizer:
 
    """
    Visualize the model performance.
    """
 
    def __init__(self):
        pass
 
    def plot_training_loss(self, history):
        """
        Plot Training Loss and Validation Loss.
        """
 
        plt.figure(figsize=(10,5))
 
        plt.plot(
            history.history["loss"],
            label="Training Loss"
        )
 
        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )
 
        plt.title("Training Loss vs Validation Loss")
 
        plt.xlabel("Epoch")
 
        plt.ylabel("Loss")
 
        plt.legend()
 
        plt.grid(True)
 
        plt.savefig("outputs/loss_curve.png")
 
        plt.show()
 
        print("Training Loss Graph Saved Successfully.")
 
    def plot_predictions(self, actual, predicted):
        """
        Plot Actual vs Predicted Passenger Counts.
        """
 
        plt.figure(figsize=(12,6))
 
        plt.plot(
            actual,
            label="Actual",
            linewidth=2
        )
 
        plt.plot(
            predicted,
            label="Predicted",
            linewidth=2
        )
 
        plt.title("Actual vs Predicted Passenger Count")
 
        plt.xlabel("Time")
 
        plt.ylabel("Passengers")
 
        plt.legend()
 
        plt.grid(True)
 
        plt.savefig("outputs/prediction.png")
 
        plt.show()
 
        print("Prediction Graph Saved Successfully.")
 
    def plot_future_forecast(self, future_values):
        """
        Plot Future Forecast.
        """
 
        plt.figure(figsize=(12,6))
 
        plt.plot(
            future_values,
            marker="o",
            linewidth=2
        )
 
        plt.title("Future Passenger Forecast")
 
        plt.xlabel("Future Months")
 
        plt.ylabel("Passengers")
 
        plt.grid(True)
 
        plt.savefig("outputs/forecast.png")
 
        plt.show()
 
        print("Forecast Graph Saved Successfully.")
 
if __name__ == "__main__":
 
    import numpy as np
 
    class DummyHistory:
 
        history = {
            "loss": [0.30,0.22,0.16,0.11,0.08],
            "val_loss":[0.35,0.26,0.20,0.15,0.10]
        }
 
    history = DummyHistory()
 
    actual = np.array([300,320,350,380,420,450])
 
    predicted = np.array([295,325,345,385,418,455])
 
    future = np.array([470,485,500,520,540,560])
 
    visualizer = Visualizer()
 
    visualizer.plot_training_loss(history)
 
    visualizer.plot_predictions(actual,predicted)
 
    visualizer.plot_future_forecast(future)
 