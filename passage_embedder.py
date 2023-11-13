from transformers import BertTokenizerFast, BertForMaskedLM
import torch.nn as nn
import torch
from numpy.linalg import norm
from numpy import dot

class _PassageEmbedder(nn.Module):
    MAX_DIM = 512

    def __init__(self, model_name):
        super(_PassageEmbedder, self).__init__()

        self.model = self._load_model(model_name)
        self.config = self.model.config

    def _load_model(self, model_name):
        if model_name != 'self': return BertForMaskedLM.from_pretrained(model_name)

    def forward(self, **inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs['hidden_states']
        last_hidden_states = hidden_states[-1]
        cls_token_embed = last_hidden_states[:, 0, :]

        return cls_token_embed

class _ModelServer:
    model = None
    tokenizer = None

    @staticmethod
    def get_model_and_tokenizer(model_name, model_ckpt, use_device):
        if _ModelServer.model is None or _ModelServer.tokenizer is None:
            _ModelServer.model = _PassageEmbedder(model_name)
            ckpt = torch.load(model_ckpt)
            _ModelServer.model.load_state_dict(ckpt)
            _ModelServer.model = _ModelServer.model.eval()
            _ModelServer.model = _ModelServer.model.to(use_device)

            _ModelServer.tokenizer = BertTokenizerFast.from_pretrained(model_name) 
        return _ModelServer.model, _ModelServer.tokenizer

def get_summary_similarity(generated_text, ground_truth_text, model_name, model_ckpt, use_device):
    model, tokenizer = _ModelServer.get_model_and_tokenizer(model_name, model_ckpt, use_device)

    generated_tokenized = tokenizer(generated_text, padding=True, return_tensors='pt')
    gt_tokenized = tokenizer(ground_truth_text, padding=True, return_tensors='pt')

    for k, v in generated_tokenized.items():
        generated_tokenized[k] = v[:, :_PassageEmbedder.MAX_DIM].to(use_device)
    for k, v in gt_tokenized.items():
        gt_tokenized[k] = v[:, :_PassageEmbedder.MAX_DIM].to(use_device)
    
    input_1_repr = model(**generated_tokenized).detach().cpu().numpy().squeeze()
    input_2_repr = model(**gt_tokenized).detach().cpu().numpy().squeeze()

    cos_sim = dot(input_1_repr, input_2_repr) / (norm(input_1_repr) * norm(input_2_repr))

    return cos_sim
