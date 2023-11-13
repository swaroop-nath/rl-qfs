from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import json
import torch.optim as optim
from allennlp.training.learning_rate_schedulers.noam import NoamLR
import torch.nn as nn
import torch

class QuerySummariser(nn.Module):
    def __init__(self, back_bone_name, max_pos_embedding):
        super(QuerySummariser, self).__init__()

        self.max_pos_embedding = max_pos_embedding
        self.back_bone = BartForConditionalGeneration.from_pretrained(back_bone_name)
        self.back_bone = self._enlarge_embed_pos(self.back_bone)
        self.config = self.back_bone.config
        self.config.max_position_embeddings = max_pos_embedding

    def _enlarge_embed_pos(self, model):
        sd = model.state_dict()

        encoder_shorter_pos_embeds = sd['model.encoder.embed_positions.weight']
        decoder_shorter_pos_embeds = sd['model.decoder.embed_positions.weight']
        new_config = model.config
        new_config.max_position_embeddings = self.max_pos_embedding

        new_model = BartForConditionalGeneration(new_config)

        correctly_shaped_encoder_pos_weight = new_model.model.encoder.embed_positions.weight.clone()
        correctly_shaped_encoder_pos_weight[:encoder_shorter_pos_embeds.shape[0]] = encoder_shorter_pos_embeds

        correctly_shaped_decoder_pos_weight = new_model.model.decoder.embed_positions.weight.clone()
        correctly_shaped_decoder_pos_weight[:decoder_shorter_pos_embeds.shape[0]] = decoder_shorter_pos_embeds

        sd['model.decoder.embed_positions.weight'] = correctly_shaped_decoder_pos_weight
        sd['model.encoder.embed_positions.weight'] = correctly_shaped_encoder_pos_weight

        new_model.load_state_dict(sd)
        return new_model

    def get_encoder(self):
        return self.back_bone.get_encoder()

    def get_decoder(self):
        return self.back_bone.get_decoder()

    def get_decoder_input_embeddings(self):
        return self.get_decoder().get_input_embeddings()

    def embed_tokens(self, input_ids):
        decoder = self.get_decoder()
        embedder = decoder.get_input_embeddings()
        scale = decoder.embed_scale
        return embedder(input_ids) * scale

    def forward(self, **inputs):
        return self.back_bone(**inputs)

    def generate(self, **inputs):
        return self.back_bone.generate(**inputs)

def _init_model_and_tokenizer(name, max_pos_embedding):
    tokenizer = BartTokenizerFast.from_pretrained(name)
    model = QuerySummariser(name, max_pos_embedding)

    return model, tokenizer

def _set_dynamic_params(ds_config, model, batch_per_gpu, grad_acc_steps):
    model_hidden_size = model.config.d_model
    ds_config['zero_optimization']['reduce_bucket_size'] = model_hidden_size * model_hidden_size
    ds_config['zero_optimization']['stage3_prefetch_bucket_size'] = 0.9 * model_hidden_size * model_hidden_size
    ds_config['zero_optimization']['stage3_param_persistence_threshold'] = 10 * model_hidden_size
    ds_config['train_micro_batch_size_per_gpu'] = batch_per_gpu
    ds_config['gradient_accumulation_steps'] = grad_acc_steps

    return ds_config

def _init_deepspeed(json_path, model, batch_per_gpu, optimizer, grad_acc_steps):
    with open(json_path, 'r') as file:
        ds_config = json.load(file)

    # effective batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * num_gpus
    ds_config = _set_dynamic_params(ds_config, model, batch_per_gpu, grad_acc_steps)

    dschf = HfDeepSpeedConfig(ds_config)
    ds_engine = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)[0]

    return ds_engine, dschf

def _init_optimizer_and_scheduler(model, optim_name, learning_rate, schedule_policy, **kwargs):
    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else: raise NotImplementedError

    if schedule_policy is not None:
        if schedule_policy == 'noam_warmup':
            scheduler = NoamLR(optimizer, model.config.d_model, warmup_steps=kwargs['warmup-steps'], factor=kwargs['noam-factor-fine-tune'])
        else: raise NotImplementedError
    else: scheduler = None

    return optimizer, scheduler
    
def get_tools(model_name, max_pos_embedding, optim_name, learning_rate, schedule_policy, ds_config_file_path, batch_per_gpu, grad_acc_steps, **kwargs):
    if not kwargs['eval-mode']:
        model, tokenizer = _init_model_and_tokenizer(model_name, max_pos_embedding)
        optimizer, scheduler = _init_optimizer_and_scheduler(model, optim_name, learning_rate, schedule_policy, **kwargs)
        ds_engine, ds_config_hf = _init_deepspeed(ds_config_file_path, model, batch_per_gpu, optimizer, grad_acc_steps)

        return {
            'ds-engine': ds_engine,
            'model': model,
            'tokenizer': tokenizer,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'ds-config-hf': ds_config_hf
        }
    else:
        model, tokenizer = _init_model_and_tokenizer(model_name, max_pos_embedding)
        return {
            'model': model,
            'tokenizer': tokenizer
        }
