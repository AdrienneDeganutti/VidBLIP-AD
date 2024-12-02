import csv
import os
import torch
from datetime import datetime

class PredictionLogger:
    def __init__(self, processor, output_csv_path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_csv_path = f"{output_csv_path}_{timestamp}_validation_predictions.csv"

        # Initialize the CSV file with headers if it does not exist
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["video_id", "prediction"])
        
        self.processor = processor

    def log(self, id, prediction):
        with open(self.output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([id, prediction])

    def log_batch(self, batch_ids, predictions, epoch):
        assert len(batch_ids) == len(predictions)

        logits = torch.max(predictions, -1)[1].data
        
        batch_decoded_texts = [
            self.processor.decode(sequence, skip_special_tokens=True).strip()
            for sequence in logits
        ]

        with open(self.output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([f'epoch: {epoch}'])
            for batch_ids, batch_decoded_texts in zip(batch_ids, batch_decoded_texts):
                writer.writerow([batch_ids, batch_decoded_texts])