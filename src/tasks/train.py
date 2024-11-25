import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, train_dataloader, optimizer, scheduler, device):
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
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Compute average loss
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss


def validate_epoch(model, val_dataloader, device):
    """
    Validate the model for one epoch.
    Args:
        model: The BLIP-2 model.
        val_dataloader: DataLoader for the validation dataset.
        device: The device for evaluation (e.g., 'cuda' or 'cpu').
    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            total_loss += loss.item()

    # Compute average loss
    avg_loss = total_loss / len(val_dataloader)
    return avg_loss


def train(model, train_dataloader, val_dataloader, processor, training_args):
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
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Device setup
    device = training_args.device
    model = model.to(device)

    # WandB initialization
    wandb.init(project="blip2-ad", name="Vision+Lang Proj Frozen",
               config=vars(training_args))

    # Training Loop
    for epoch in range(training_args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")

        # Train
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validate
        avg_val_loss = validate_epoch(model, val_dataloader, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Log average losses to WandB
        wandb.log({"train_loss": avg_train_loss, "validation_loss": avg_val_loss})

    # Save the final model
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("Model and processor saved!")
