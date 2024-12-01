import torch
import wandb

from eval_metrics.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from eval_metrics.pycocoevalcap.bleu.bleu import Bleu
from eval_metrics.pycocoevalcap.meteor.meteor import Meteor
from eval_metrics.pycocoevalcap.rouge.rouge import Rouge
from eval_metrics.pycocoevalcap.cider.cider import Cider

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
    

    def compute_metrics(self, outputs, batch):

        logits = outputs['logits']  # Shape: [batch, seq, dim]
        predicted_tokens = torch.argmax(logits, dim=-1)  # Shape: [batch, seq]

        # Convert references to tokens
        references = batch['labels']  # Shape: [batch, seq]

        # Detach and convert tensors to lists
        predicted_tokens = predicted_tokens.cpu().tolist()
        references = references.cpu().tolist()

        # Prepare data for pycocoevalcap
        res = {}
        gts = {}

        for idx, (pred, ref) in enumerate(zip(predicted_tokens, references)):
            # Filter out padding tokens
            pred = [token for token in pred if token != -100]
            ref = [token for token in ref if token != -100]

            # Decode token indices to text
            pred_text = self.processor.decode(pred, skip_special_tokens=True).strip()
            ref_text = self.processor.decode(ref, skip_special_tokens=True).strip()

            # Populate result and ground truth dictionaries
            res[idx] = [{"caption": pred_text}]
            gts[idx] = [{"caption": ref_text}]

        # Tokenize using PTBTokenizer
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # Initialize scorers
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # Compute scores
        scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):  # BLEU returns multiple scores
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score

        return scores