import csv
import os
from os.path import join
import torch
from datetime import datetime

class PredictionLogger:
    def __init__(self, processor, output_dir):
        self.processor = processor
        self.output_dir = output_dir
    
    def write_csv(self, epoch):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_prediction_filepath = join(self.output_dir, f'epoch_{epoch}', f'{timestamp}_predictions.csv')
        
        os.makedirs(join(self.output_dir, f'epoch_{epoch}'), exist_ok=True)
        with open(output_prediction_filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["video_id", "prediction"])

        return output_prediction_filepath


    def log(self, id, prediction, epoch):
        logits = torch.max(prediction, -1)[1].data
        decoded_text = self.processor.decode(logits[0], skip_special_tokens=True).strip()

        output_prediction_filepath = self.write_csv(epoch)
        with open(output_prediction_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([id, decoded_text])


    def log_batch(self, batch_ids, predictions, epoch):
        assert len(batch_ids) == len(predictions)

        logits = torch.max(predictions, -1)[1].data
        
        batch_decoded_texts = [
            self.processor.decode(sequence, skip_special_tokens=True).strip()
            for sequence in logits
        ]

        output_prediction_filepath = self.write_csv(epoch)
        with open(output_prediction_filepath, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([f'epoch: {epoch}'])
            for batch_ids, batch_decoded_texts in zip(batch_ids, batch_decoded_texts):
                writer.writerow([batch_ids, batch_decoded_texts])