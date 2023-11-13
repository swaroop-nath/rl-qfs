import torch
import torch.nn as nn
from torch.distributions import Categorical

def run_batch(ds_engine, batch, **kwargs):
    labels = batch.pop('labels')

    is_scheduled_sample = kwargs.get('sampling-type', None)
    if is_scheduled_sample is not None and is_scheduled_sample == 'scheduled-sample':
        return run_batch_scheduled_sample(ds_engine, batch, labels, **kwargs)

    logits = ds_engine.module(**batch)['logits']
    loss = compute_loss(logits, labels, **kwargs)

    return {'loss': loss, 'logits': logits}

def run_batch_scheduled_sample(ds_engine, batch, labels, **kwargs):
    # 1. First run it for k-1 passes
    # Note that after every passes you have to take the [:-1] words
    # and prepend the start of token to them
    enc_inputs = {'input_ids': batch.pop('input_ids'), 'attention_mask': batch['attention_mask']}
    encoder = ds_engine.get_encoder()
    encoder_outputs = encoder(**enc_inputs, return_dict=True)
    batch['encoder_outputs'] = encoder_outputs

    for _ in range(kwargs['num-dec-pass'] - 1):
        if kwargs['last-pass-grad-only']:
            with torch.no_grad():
                logits = ds_engine.module(**batch)['logits']
                logits = logits.to(torch.float32)
                all_but_penultimate_logits = logits[:, :-1, :]
                distribution = Categorical(logits=all_but_penultimate_logits)

                actions = distribution.sample()
                batch['decoder_input_ids'][:, 1:] = actions
                
        else:
            if 'decoder_input_ids' in batch.keys():
                decoder_input_ids = batch.pop('decoder_input_ids')
                decoder_inputs_embeds = ds_engine.module.embed_tokens(decoder_input_ids)
                batch['decoder_inputs_embeds'] = decoder_inputs_embeds

            logits = ds_engine.module(**batch)['logits']
            all_but_penultimate_logits = logits[:, :-1, :]
            rand_adder = torch.rand(all_but_penultimate_logits.size()[-1]).to(all_but_penultimate_logits.get_device()) # G_y only for vocab words, doesn't vary across time/batch
            # Ref - Goyal, 2017: https://arxiv.org/pdf/1704.06970.pdf
            all_but_penultimate_logits = (all_but_penultimate_logits + rand_adder) / kwargs['scheduled-sampling-temp']

            all_but_penultimate_proba = torch.softmax(all_but_penultimate_logits, dim=-1) # (batch_size, seq_len, vocab_size)
            embeddings = ds_engine.module.get_decoder_input_embeddings().weight.clone().detach()
            all_but_penultimate_logits = all_but_penultimate_proba.to(torch.float32)
            embeddings = embeddings.to(torch.float32)
            new_decoder_input_embeds = torch.tensordot(all_but_penultimate_proba, embeddings, dims=([-1], [0]))
            batch['decoder_inputs_embeds'][:, 1:] = new_decoder_input_embeds

    logits = ds_engine.module(**batch)['logits']
    loss = compute_loss(logits, labels, **kwargs)

    return {'loss': loss, 'logits': logits}

def compute_loss(batch_pred, batch_true, **kwargs):
    # Computing loss for only the masked tokens
    # batch_pred.shape == (batch_size, seq_len, vocab_size)
    # batch_true.shape == (batch_size, seq_len)
    preds = batch_pred.reshape(-1, batch_pred.size(-1)) # (batch_size * seq_len, vocab_size)
    true = batch_true.reshape(-1).type(torch.LongTensor).to(preds.get_device()) # (batch_size * seq_len)

    return nn.functional.cross_entropy(input=preds, target=true, \
            ignore_index=kwargs['pad-token-index'], label_smoothing=kwargs['cross-entropy-label-smoothing-factor'])