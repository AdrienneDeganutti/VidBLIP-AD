import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerModel
)
from src.modeling.Blip2ForConditionalGeneration import Blip2ForConditionalGeneration
from src.modeling.Blip2QFormer import Blip2QFormerModel


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config, training_args) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.vision_projection = nn.Linear(in_features=1536, out_features=1408)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()
