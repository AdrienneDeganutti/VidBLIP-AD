import torch
import wandb

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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

        bleu_scores = self.BLEU(predicted_tokens, references)
        rouge_score = self.ROUGE(predicted_tokens, references)


    def BLEU(self, predicted_tokens, references):

        weights = [                     # Calculate BLEU-1, BLEU-2, BLEU-3 and BLEU-4 simultaneously
            (1.,),
            (1./2., 1./2.),
            (1./3., 1./3., 1./3.),
            (1./4., 1./4., 1./4., 1./4.)
        ]

        bleu_scores = {f"bleu-{i+1}": [] for i in range(len(weights))}

        for pred, ref in zip(predicted_tokens, references):
            # Filter out padding tokens (assumes padding token is -100)
            pred = [token for token in pred if token != -100]
            ref = [token for token in ref if token != -100]

            for i, weight in enumerate(weights):
                score = sentence_bleu(
                    [ref], pred,
                    weights=weight,
                    smoothing_function=SmoothingFunction().method1
                )
                bleu_scores[f"bleu-{i+1}"].append(score)

        # Average BLEU scores across the batch
        avg_bleu_scores = {key: sum(values) / len(values) for key, values in bleu_scores.items()}

        return avg_bleu_scores
    

    def ROUGE(self, predicted_tokens, references):
         # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Accumulator for ROUGE-L scores
        rouge_l_scores = []

        for pred, ref in zip(predicted_tokens, references):
            # Filter out padding tokens (assumes padding token is 0)
            pred = [token for token in pred if token != 0]
            ref = [token for token in ref if token != 0]

            # Convert token indices to strings for ROUGE computation
            pred_text = " ".join(map(str, pred))
            ref_text = " ".join(map(str, ref))

            # Compute ROUGE-L score
            score = scorer.score(ref_text, pred_text)
            rouge_l_scores.append(score["rougeL"].fmeasure)

        # Average ROUGE-L score across the batch
        avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)

        return {"rougeL": avg_rouge_l_score}

