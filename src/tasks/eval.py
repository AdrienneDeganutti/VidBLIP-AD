import torch
import wandb


class Evaluation:
    def __init__(self, training_args, model, processor, val_dataloader, train_dataloader):
        self.args = training_args
        self.model = model
        self.processor = processor 
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader

    def evaluate(self, output_dir, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                pixel_values = batch["pixel_values"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                total_loss += loss.item()

                wandb.log({"val_step_loss": loss})
        
            #logits = torch.max(outputs.logits, -1)[1].data
            #decoded_text = self.processor.decode(logits, skip_special_tokens=True).strip()
            #print(decoded_text)

        # Compute average loss
        avg_loss = total_loss / len(self.val_dataloader)
        avg_acc = 1
        return avg_acc, avg_loss
    
    def compute_metrics(self, outputs):

        step_accuracy = {bleu, rouge, meteor, cider}

        return step_accuracy

