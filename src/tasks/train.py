import sys
import os

from collections.abc import Callable
from functools import partial
from typing import Any

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

import torch
from transformers import Blip2Processor

from src.datasets.frame import FrameDataset
from src.datasets.utils import (
    DataCollatorForVideoSeq2Seq,
    clean_narration_text,
    generate_input_ids_and_labels,
)
from src.modeling.model_blip import VideoBlipForConditionalGeneration
from configs.load_config import get_custom_args

from src.modeling.trainer import Trainer

PROMPT = "Question: What is the camera wearer doing? Answer:"


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

    return preprocessed




def train():
    config_file = "configs/training_config.json"
    model_args, data_args, training_args = get_custom_args(config_file)

    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False

    processor = Blip2Processor.from_pretrained(
        model_args.model_name_or_path
    )
    model = VideoBlipForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        training_args,
        #low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
        low_cpu_mem_usage=False
    )
    
    # freeze everything except for qformer and vision_model:
    for param in model.language_model.parameters():
        param.requires_grad = False

    # ensure that the vision model parameters are trainable:
    #for param in model.vision_model.parameters():
    #    param.requires_grad = True

    # we need to enable input require grads since the vision model (the first layer) is frozen.
    model.enable_input_require_grads()
    model = model.to(training_args.device)

    train_data = FrameDataset(
        model_args,
        data_args.train_visual_features_dir,
        data_args.train_annotation_file,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model
        ),
    )
    val_data = FrameDataset(
        model_args,
        data_args.val_visual_features_dir,
        data_args.val_annotation_file,
        transform=partial(
            preprocess,
            processor,
            decoder_only_lm=model.config.use_decoder_only_language_model
        ),
    )

    # Load the best model at the end so we can save it
    training_args.load_best_model_at_end = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()