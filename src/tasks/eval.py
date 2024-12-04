import torch
import wandb
from src.utils.log_predictions import PredictionLogger
from eval_metrics.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from eval_metrics.pycocoevalcap.bleu.bleu import Bleu
from eval_metrics.pycocoevalcap.meteor.meteor import Meteor
from eval_metrics.pycocoevalcap.rouge.rouge import Rouge
from eval_metrics.pycocoevalcap.cider.cider import Cider
from src.utils.logger import LOGGER as logger


class Evaluation:
    def __init__(self, training_args, logging_args, model, processor, val_dataloader, train_dataloader):
        self.args = training_args
        self.logging = logging_args
        self.model = model
        self.processor = processor 
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        
        self.log_predictions = PredictionLogger(processor, training_args.output_dir)


    def evaluate(self, output_dir, epoch):
        self.model.eval()
        epoch_loss = 0.0

        if self.args.include_for_metrics:
            batch_count = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                video_id = batch["filenames"]
                batch = {key: value.to(self.args.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
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
                epoch_loss += loss.item()

                wandb.log({"validation_loss_batch": loss}) if self.logging.wandb_log else None

                if batch_count == 0 and epoch % self.args.eval_steps == 0 and epoch != 0:
                    val_batch_metrics = self.compute_metrics(outputs, batch)
                    metric_names = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR', 'CIDEr']
                    log_metrics = {f"val_batch_{name}": val_batch_metrics[name] for name in metric_names}
                    wandb.log(log_metrics) if self.logging.wandb_log else None
                    batch_count += 1
                
                    if self.args.per_device_train_batch_size == 1:
                        self.log_predictions.log(video_id, outputs['logits'], epoch)
                    elif self.args.per_device_train_batch_size > 1:
                        self.log_predictions.log_batch(video_id, outputs['logits'], epoch)


        # Compute average loss
        avg_loss = epoch_loss / len(self.val_dataloader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        wandb.log({"validation_loss_epoch": avg_loss}) if self.logging.wandb_log else None

        return
    

    def compute_metrics(self, outputs, batch):

        logits = outputs['logits']  # Shape: [batch, seq, dim]
        predicted_tokens = torch.argmax(logits, dim=-1)  # Shape: [batch, seq]

        references = batch['labels']  # Shape: [batch, seq]

        predicted_tokens = predicted_tokens.cpu().tolist()
        references = references.cpu().tolist()

        res = {}
        gts = {}

        for idx, (pred, ref) in enumerate(zip(predicted_tokens, references)):
            pred = [token for token in pred if token != -100]
            ref = [token for token in ref if token != -100]

            pred_text = self.processor.decode(pred, skip_special_tokens=True).strip()
            ref_text = self.processor.decode(ref, skip_special_tokens=True).strip()

            res[idx] = [{"caption": pred_text}]
            gts[idx] = [{"caption": ref_text}]

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):  # BLEU returns multiple scores
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score

        return scores
