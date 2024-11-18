import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor
from torch.optim import AdamW
from functools import partial


def train(model, train_dataloader, val_dataloader, processor, training_args):
    """
    Args:
        model: The BLIP-2 model.
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        processor: The Blip2Processor for pre-processing.
        training_args: Training arguments (e.g., learning rate, epochs).
    """
    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    device = training_args.device
    model = model.to(device)
    
    # Training Loop
    for epoch in range(training_args.num_train_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Adjust if different label format is needed
            )
            loss = outputs.loss
            train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if (batch_idx + 1) % training_args.logging_steps == 0:
                print(f"Epoch {epoch+1}/{training_args.num_train_epochs}, Step {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                val_loss += outputs.loss.item()

        print(f"Epoch {epoch+1} Summary: Train Loss = {train_loss/len(train_dataloader)}, Val Loss = {val_loss/len(val_dataloader)}")

    # Save the final model
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("Model and processor saved!")