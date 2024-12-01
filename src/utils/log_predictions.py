import csv
import os
from datetime import datetime

class PredictionLogger:
    def __init__(self, output_csv_path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_csv_path = f"{output_csv_path}_{timestamp}_validation_predictions.csv"

        # Initialize the CSV file with headers if it does not exist
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "reference", "prediction"])

    def log(self, id, reference, prediction):
        """
        Logs a single prediction to the CSV file.

        Args:
            id (str): Unique identifier for the example.
            reference (str): The reference text (ground truth).
            prediction (str): The predicted text.
        """
        with open(self.output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([id, reference, prediction])

    def log_batch(self, batch_ids, references, predictions):
        """
        Logs a batch of predictions to the CSV file.

        Args:
            batch_ids (list of str): List of unique identifiers for each example in the batch.
            references (list of str): List of reference texts (ground truths) for the batch.
            predictions (list of str): List of predicted texts for the batch.
        """
        assert len(batch_ids) == len(references) == len(predictions), (
            "Batch lengths for IDs, references, and predictions must match."
        )

        with open(self.output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for example_id, reference, prediction in zip(batch_ids, references, predictions):
                writer.writerow([example_id, reference, prediction])

# Example usage
# logger = PredictionLogger("predictions.csv")
# logger.log("example_1", "This is the reference.", "This is the prediction.")
# logger.log_batch(["example_2", "example_3"], ["Ref 1", "Ref 2"], ["Pred 1", "Pred 2"])