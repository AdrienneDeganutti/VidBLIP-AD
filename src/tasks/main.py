import sys
import os
from functools import partial
from typing import Any

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

import torch
import psutil
from transformers import Blip2Processor
from src.utils.logger import LOGGER as logger
from src.datasets.frame import FrameDataset
from src.datasets.utils import (
    DataCollatorForVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels,
)
from src.modeling.model_blip import VideoBlipForConditionalGeneration
from configs.load_config import get_custom_args
from torch.utils.data import DataLoader

from src.tasks.train import Train

PROMPT = "Please provide a detailed description of this movie clip."

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


def preprocess(
            processor: Blip2Processor,
            item: dict[str, Any],
            decoder_only_lm: bool = True
        ) -> dict[str, torch.Tensor]:
    
    # tokenize text inputs
    cleaned_narration_text = clean_narration_text(item["captions"])
    preprocessed = generate_input_ids_and_labels(
        processor.tokenizer, PROMPT, cleaned_narration_text, decoder_only_lm
    )
    preprocessed["pixel_values"] = item["frames"]
    preprocessed['video_id'] = item['cmd_ad_filename']

    return preprocessed


def main():
    config_file = "configs/training_config.json"
    model_args, data_args, training_args, logging_args = get_custom_args(config_file)
    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False

    processor = Blip2Processor.from_pretrained(model_args.model_name_or_path)
    model = VideoBlipForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        training_args,
        low_cpu_mem_usage=False
    )
    
    # freeze everything except for qformer & vision projection head:
    for param in model.language_model.parameters():
        param.requires_grad = False
    for param in model.qformer.parameters():
        param.requires_grad = True
    for param in model.vision_projection.parameters():
        param.requires_grad = True
    for param in model.language_projection.parameters():
        param.requires_grad = True
    # we need to enable input require grads since the vision model (the first layer) is frozen.
    model.enable_input_require_grads()
    logger.info('Moving model to device...')
    model = model.to(training_args.device)


    logger.info('Loading training dataset...')
    train_data = FrameDataset(
        model_args=model_args,
        frames_dir=data_args.train_visual_features_dir,
        annotation_file=data_args.train_annotation_file,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model
        ),
    )
    print_cpu_memory()
    
    logger.info('Loading validation dataset...')
    val_data = FrameDataset(
        model_args=model_args,
        frames_dir=data_args.val_visual_features_dir,
        annotation_file=data_args.val_annotation_file,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model
        ),
    )
    print_cpu_memory()

    training_args.load_best_model_at_end = True

    data_collator = DataCollatorForVideoSeq2Seq(
        processor.tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    training_args.max_iter = len(train_dataloader)

    val_dataloader = DataLoader(
        val_data,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    trainer = Train(training_args, logging_args, model, processor, val_dataloader, train_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()