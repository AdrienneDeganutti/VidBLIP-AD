import torch
import wandb
import time
from datetime import timedelta
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.tasks.eval import Evaluation
from src.utils.logger import LOGGER as logger


def mixed_precision_init(training_args, model, train_dataloader):
    # Optimizer and Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

    return optimizer, num_training_steps, scheduler


def train_epoch(training_args, model, train_dataloader, optimizer, scheduler):
    """
    Train the model for one epoch.
    Args:
        model: The BLIP-2 model.
        train_dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for gradient updates.
        scheduler: Learning rate scheduler.
        device: The device for training (e.g., 'cuda' or 'cpu').
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0


    for batch in train_dataloader:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        total_loss += loss.item()
        wandb.log({"train_step_loss": loss})

        if training_args.include_for_metrics:
            train_step_acc = Evaluation.compute_metrics(outputs)


        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Compute average loss
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss


def train(model, processor, train_dataloader, val_dataloader, training_args):
    """
    Train and validate the BLIP-2 model.
    Args:
        model: The BLIP-2 model.
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        processor: The Blip2Processor for pre-processing.
        training_args: Training arguments (e.g., learning rate, epochs).
    """
    # Optimizer and Scheduler
    optimizer, num_training_steps, scheduler = mixed_precision_init(training_args, model, train_dataloader)

    # WandB initialization
    wandb.init(project="blip2-ad", name="Debugging",
               config=vars(training_args))
    
    if training_args.do_eval:
        evaluation = Evaluation(training_args, model, processor, val_dataloader, train_dataloader)
    
    start_training_time = time.time()

    # Training Loop
    for epoch in range(training_args.num_train_epochs):

        if epoch == 0 & training_args.eval_on_start & training_args.do_eval:
            logger.info('Performing initial validation at epoch 0.')
            epoch_0_val_acc, epoch_0_val_loss = evaluation.evaluate(training_args.output_dir, epoch)
            logger.info(f"Epoch 0 Validation Accuracy: {epoch_0_val_acc}")
            wandb.log({"validation_loss": epoch_0_val_loss})

        logger.info(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")

        avg_train_loss = train_epoch(training_args, model, train_dataloader, optimizer, scheduler)
        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        if training_args.do_eval:       #evaluate at every epoch
            avg_val_acc, avg_val_loss = evaluation.evaluate(training_args.output_dir, epoch)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        wandb.log({"train_loss": avg_train_loss, "validation_loss": avg_val_loss,
                   "train_accuracy": avg_train_acc, "validation_accuracy": avg_val_acc})
    

    total_training_time = time.time() - start_training_time
    total_time_str = str(timedelta(seconds=total_training_time))
    logger.info(f'Total training time: {total_time_str} ({(total_training_time / training_args.max_iter):.4f} s / iter)')

    wandb.finish()


    # Save the final model
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("Model and processor saved!")
