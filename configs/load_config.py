from dataclasses import dataclass, field
import json
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str
    num_subsample_frames: int

@dataclass 
class DataArguments:
    train_visual_features_dir: str
    val_visual_features_dir: str
    train_annotation_file: str
    val_annotation_file: str

@dataclass
class TrainingArguments(HFTrainingArguments):
    optim: str = field(default="adamw_torch")


def get_custom_args(config_file: str):

    with open(config_file, 'r') as f:
        config = json.load(f)

    model_args = ModelArguments(
        model_name_or_path=config['model_name_or_path'],
        num_subsample_frames=config['num_subsample_frames']
    )

    data_args = DataArguments(
        train_visual_features_dir=config['train_visual_features_dir'],
        val_visual_features_dir=config['val_visual_features_dir'],
        train_annotation_file=config['train_annotation_file'],
        val_annotation_file=config['val_annotation_file']
    )

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        do_train=config['do_train'],
        do_eval=config['do_eval'],
        num_train_epochs=config['num_train_epochs'],
        warmup_steps=config['warmup_steps'],
        learning_rate=config['learning_rate'],
        
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        weight_decay=config['weight_decay'],
        dataloader_num_workers=config['dataloader_num_workers'],
        bf16=config['bf16'],
        eval_on_start=config['eval_on_start'],
        include_inputs_for_metrics=config['include_inputs_for_metrics'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        eval_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        logging_steps=config['logging_steps']
    )

    return model_args, data_args, training_args