import torch
import wandb
import time
from datetime import timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.tasks.eval import Evaluation
from src.utils.logger import LOGGER as logger


class Train:
    def __init__(self, training_args, model, processor, val_dataloader, train_dataloader):
        self.args = training_args
        self.model = model
        self.processor = processor 
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader

        if self.args.do_eval:
            self.evaluation = Evaluation(self.args, self.model, self.processor, self.val_dataloader, self.train_dataloader)


    def mixed_precision_init(self):
        # Optimizer and Scheduler
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

        return optimizer, num_training_steps, scheduler


    def train_epoch(self, optimizer, scheduler):

        self.model.train()
        total_loss = 0.0


        for batch in self.train_dataloader:
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
            wandb.log({"train_step_loss": loss})

            if self.args.include_for_metrics:
                train_step_acc = self.evaluation.compute_metrics(outputs, batch)


            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Compute average loss
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss


    def train(self):

        # Optimizer and Scheduler
        optimizer, num_training_steps, scheduler = self.mixed_precision_init()

        # WandB initialization
        wandb.init(project="blip2-ad", name="Debugging",
               config=vars(self.args))
    
        start_training_time = time.time()

        # Training Loop
        for epoch in range(self.args.num_train_epochs):

            if epoch == 0 & self.args.eval_on_start & self.args.do_eval:
                logger.info('Performing initial validation at epoch 0.')
                epoch_0_val_acc, epoch_0_val_loss = self.evaluation.evaluate(self.args.output_dir, epoch)
                logger.info(f"Epoch 0 Validation Accuracy: {epoch_0_val_acc}")
                wandb.log({"validation_loss": epoch_0_val_loss})

            logger.info(f"Epoch {epoch + 1}/{self.args.num_train_epochs}")

            avg_train_loss = self.train_epoch(optimizer, scheduler)
            logger.info(f"Train Loss: {avg_train_loss:.4f}")

            if self.args.do_eval:       #evaluate at every epoch
                avg_val_acc, avg_val_loss = self.evaluation.evaluate(self.args.output_dir, epoch)
                logger.info(f"Validation Loss: {avg_val_loss:.4f}")

            wandb.log({"train_loss": avg_train_loss, "validation_loss": avg_val_loss,
                        "train_accuracy": avg_train_acc, "validation_accuracy": avg_val_acc})
    

        total_training_time = time.time() - start_training_time
        total_time_str = str(timedelta(seconds=total_training_time))
        logger.info(f'Total training time: {total_time_str} ({(total_training_time / self.args.max_iter):.4f} s / iter)')

        wandb.finish()


        # Save the final model
        self.model.save_pretrained(self.args.output_dir)
        self.processor.save_pretrained(self.args.output_dir)
        print("Model and processor saved!")
