import torch
import torch.nn as nn
from torch.distributions import Categorical
from numpy.random import randint, uniform
from utils import get_text_similarity
from copy import deepcopy

def run_batch(ds_engine, batch, **kwargs):
    # print('Running PG')
    labels = batch.pop('labels')
    tokenizer = kwargs['tokenizer']

    mle_gen = kwargs['mle-gen']
    rl_preds, rl_log_probs = run_batch_rl(ds_engine, batch, mle_gen, labels, **kwargs)

    mle_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in mle_gen]
    rl_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in rl_preds]
    target_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in labels]

    # Reward gen is costly
    return_value = reward_generation(rl_summary, target_summary, rl_log_probs.get_device(), **kwargs)
    baseline_value = reward_generation(mle_summary, target_summary, rl_log_probs.get_device(), **kwargs)

    rl_loss = -(return_value - baseline_value) * rl_log_probs
    rl_loss = torch.mean(rl_loss)

    if rl_loss < 0.0: rl_loss = torch.tensor(0.0, device=rl_loss.get_device())

    return {'loss': rl_loss, 'marginal_reward': torch.mean(return_value - baseline_value), 'reward': torch.mean(return_value)}

def run_batch_rl(ds_engine, batch, mle_gen, labels, **kwargs):
    if kwargs['sampling-type'] == 'rl-window': 
        rl_preds, log_probs = run_batch_rl_window(ds_engine, batch, mle_gen, labels, **kwargs)
        return rl_preds, log_probs
    elif kwargs['sampling-type'] == 'scheduled-sample': 
        rl_preds, log_probs = run_batch_rl_scheduled(ds_engine, batch, labels, mle_gen_rl_scheduled=False, **kwargs)
        return rl_preds, log_probs

def run_batch_rl_window(ds_engine, batch, mle_gen, labels, **kwargs):
    # Refining mle_gen by prepending decoder start token in
    prepend_vector = torch.ones((mle_gen.size(0), 1), dtype=mle_gen.dtype, device=mle_gen.get_device()) * kwargs['decoder-start-token-id']
    mle_gen = torch.cat((prepend_vector, mle_gen), dim=1)
    labels = torch.cat((prepend_vector, labels), dim=1)

    # There should be atleast 1 token, hence 1 rather than 0
    # Generating an upto towards the very end makes the batch inefficient for training,
    # most of the log-proba would be multiplied by zero
    
    if mle_gen.size(1) > (kwargs['rl-window'] + 1): upto = randint(1, mle_gen.size(1) - kwargs['rl-window'])
    else: upto = 1

    rl_decoder_input_ids = mle_gen[:, :upto]
    batch['decoder_input_ids'] = rl_decoder_input_ids[:, -kwargs['context-window']:]

    enc_inputs = {'input_ids': batch.pop('input_ids'), 'attention_mask': batch['attention_mask']}
    encoder = ds_engine.get_encoder()
    encoder_outputs = encoder(**enc_inputs, return_dict=True)
    batch['encoder_outputs'] = encoder_outputs

    log_probs = []
    all_actions = []
    for di in range(min(kwargs['rl-window'], labels[:, upto: upto + kwargs['rl-window']].size(1))):
        all_logits = ds_engine.module(**batch)['logits']
        next_token_logits = all_logits[:, -1, :]
        # next_token_proba = torch.softmax(next_token_logits, dim=1)

        # Taking an action
        dtype = next_token_logits.dtype
        next_token_logits = next_token_logits.to(torch.float32) 
        distribution = Categorical(logits=next_token_logits)
        if uniform() <= kwargs['prof-force-proba'] and (upto + di) < labels.size(1): 
            action = labels[:, upto + di]
        else: 
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = log_prob.to(dtype)
        log_probs.append(log_prob)

        # Forming the new decoder input
        rl_decoder_input_ids = batch['decoder_input_ids']
        action = action.detach()
        rl_decoder_input_ids = torch.cat((rl_decoder_input_ids, action[:, None]), dim=1)
        batch['decoder_input_ids'] = rl_decoder_input_ids[:, -kwargs['context-window']:]

        # Storing the action
        all_actions.append(action)

    all_actions = torch.stack(all_actions, dim=1)
    rl_preds = torch.cat((mle_gen[:, 1:upto], all_actions), dim=1) # discarding the decoder start token id
    # upto + rl-window --> from RL gen
    if upto + kwargs['rl-window'] < mle_gen.size(1):
        rl_preds = torch.cat((rl_preds, mle_gen[:, upto + kwargs['rl-window']:]), dim=1)
    mle_gen = mle_gen[:, 1:] # discarding the decoder start token id

    # Getting the avg log probs
    log_probs = torch.stack(log_probs, dim=1)
    log_probs_mask = 1 - torch.eq(labels[:, upto: upto + kwargs['rl-window']], kwargs['pad-token-index']).type(torch.int32).to(log_probs.get_device())
    log_probs = log_probs * log_probs_mask
    lens = torch.sum(log_probs_mask, dim=1)
    log_probs = torch.sum(log_probs, dim=1) / (lens + 0.1)

    return rl_preds, log_probs

def run_batch_rl_scheduled(ds_engine, batch, labels, mle_gen_rl_scheduled=False, **kwargs):
    # 1. First run it for k-1 passes
    # Note that after every passes you have to take the [:-1] words
    # and prepend the start of token to them
    enc_inputs = {'input_ids': batch.pop('input_ids'), 'attention_mask': batch['attention_mask']}
    encoder = ds_engine.get_encoder()
    encoder_outputs = encoder(**enc_inputs, return_dict=True)
    batch['encoder_outputs'] = encoder_outputs

    for _ in range(kwargs['num-dec-pass'] - 1):
        if kwargs['last-pass-grad-only'] or mle_gen_rl_scheduled:
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

    if mle_gen_rl_scheduled: 
        with torch.no_grad():
            logits = ds_engine.module(**batch)['logits']
            return torch.argmax(logits, dim=-1)

    logits = ds_engine.module(**batch)['logits']
    logits_dtype = logits.dtype
    logits = logits.to(torch.float32)
    dist = Categorical(logits=logits)
    actions = dist.sample() # (batch_size, seq_len)
    log_probs = dist.log_prob(actions) # (batch_size, seq_len)
    log_probs = log_probs.to(logits_dtype)
    log_probs_mask = 1 - torch.eq(labels, kwargs['pad-token-index']).type(torch.int32).to(log_probs.get_device())
    log_probs = log_probs * log_probs_mask
    lens = torch.sum(log_probs_mask, dim=1)
    log_probs = torch.sum(log_probs, dim=1) / (lens + 0.1)

    return actions, log_probs

def reward_generation(pred_summaries, target_summaries, device, **kwargs):
    scores = []
    for i in range(len(pred_summaries)):
        score = get_text_similarity(pred_summaries[i], target_summaries[i], **kwargs)
        scores.append(score)

    r_l = torch.FloatTensor(scores).to(device)

    return r_l