import os
import torch
import psutil
import wandb
import time
from os.path import join
from datetime import timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.tasks.eval import Evaluation
from src.utils.logger import LOGGER as logger


def print_cpu_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU memory used: {memory_info.rss / 1024 ** 2:.2f} MB")  # rss is the resident set size

def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Convert bytes to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Convert bytes to MB
        print(f"GPU memory allocated: {gpu_memory_allocated:.2f} MB")
        print(f"GPU memory reserved: {gpu_memory_reserved:.2f} MB")
    else:
        print("No GPU available.")


class Train:
    def __init__(self, training_args, logging_args, model, processor, val_dataloader, train_dataloader):
        self.args = training_args
        self.logging = logging_args
        self.model = model
        self.processor = processor 
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.evaluation = Evaluation(self.args, self.logging, self.model, self.processor, self.val_dataloader, self.train_dataloader)


    def mixed_precision_init(self):

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

        return optimizer, scheduler
    

    def save_model(self, epoch):
        checkpoint_dir = join(self.args.output_dir, f'epoch_{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        return


    def train_epoch(self, optimizer, scheduler, epoch):

        self.model.train()
        epoch_loss = 0.0

        if self.args.include_for_metrics:
            #total_metrics = {name: 0.0 for name in ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR', 'CIDEr']}
            batch_count = 0


        for batch in self.train_dataloader:
            video_id = batch["filenames"]
            batch = {key: value.to(self.args.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]              # PROMPT + label + eos
            attention_mask = batch["attention_mask"]    # length of PROMPT + label + eos
            labels = batch["labels"]                    # Masked PROMPT + label + eos

            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            epoch_loss += loss.item()
            wandb.log({"training_loss_step": loss}) if self.logging.wandb_log else None

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if batch_count == 0 and epoch % 3 == 0 and epoch != 0:      # Only perform evaluation on the first batch of every 3 epochs
                train_batch_metrics = self.evaluation.compute_metrics(outputs, batch)
                metric_names = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR', 'CIDEr']
                log_metrics = {f"train_batch_{name}": train_batch_metrics[name] for name in metric_names}
                wandb.log(log_metrics)
                batch_count += 1

        # Compute average loss over the full epoch
        avg_loss = epoch_loss / len(self.train_dataloader)
        logger.info(f"Training Loss: {avg_loss:.4f}")
        wandb.log({"training_loss_epoch": avg_loss}) if self.logging.wandb_log else None
        
        return


    def train(self):
        optimizer, scheduler = self.mixed_precision_init()

        if self.logging.wandb_log:
            wandb.login() 
            wandb.init(project="VidBLIP-AD", name=self.logging.wandb_project,
                        config=vars(self.args))
    
        start_training_time = time.time()

        for epoch in range(self.args.num_train_epochs):
            
            logger.info(f"Epoch {epoch}/{self.args.num_train_epochs}")
            self.train_epoch(optimizer, scheduler, epoch)
            self.evaluation.evaluate(self.args.output_dir, epoch)

            if epoch != 0 and epoch % self.logging.checkpoint_epoch == 0:
                self.save_model(epoch)
                logger.info(f"Model and processor saved to {self.args.output_dir}epoch_{epoch}.")
    
        total_training_time = time.time() - start_training_time
        total_time_str = str(timedelta(seconds=total_training_time))
        logger.info(f'Total training time: {total_time_str} ({(total_training_time / self.args.max_iter):.4f} s / iter)')

        # Save the final model
        self.save_model(epoch)
        logger.info("Model and processor saved!")

        wandb.finish() if self.logging.wandb_log else None
