import torch
from model_code import get_tools
from config import ConfigManager
from data_handler import get_s2s_dataloader
from tqdm import tqdm
from utils import get_rouge_score
import numpy as np
import pandas as pd

progress_log_file = open('progress_eval_b.log', 'w')
initial_pos = progress_log_file.tell()
DEVICE = 'cuda:0'

def _evaluate_trained_model(trained_model, tokenizer, data_loader, **kwargs):
    num_eval_batches = len(data_loader)
    avg_scores = {}

    greedy_kwargs = {'min_length': 64, 'max_length': 256}
    beam_kwargs = {'min_length': 64, 'max_length': 256, 'num_beams': 15, 'no_repeat_ngram_size': 2} # Better results
    # beam_kwargs = {'min_length': 64, 'max_length': 256, 'num_beams': 5, 'no_repeat_ngram_size': 3} # Usually beam works better, much better
    top_p_top_k_kwargs = {'min_length': 64, 'max_length': 256, 'do_sample': True, 'temperature': 1, 'top_p': 0.8, 'no_repeat_ngram_size': 3, 'num_beams': 5, 'top_k': 100}

    gen_kwargs = beam_kwargs

    # Doing next(iter(data_loader)) without shuffle loads the same sample
    # As in every step, iter(data_loader) creates a new iterator, that starts from 0
    # This is the same problem with training, but as in training
    # it seems samples for a lot of times, it doesn't really matter.
    generator = iter(data_loader)

    if kwargs['store-gen']: 
        template = {'query_context': [], 'actual_summary': [], 'pred_summary': []}
    p_bar = tqdm(total=num_eval_batches, desc='Evaluating', leave=True, file=progress_log_file)
    for step in range(num_eval_batches):
        progress_log_file.seek(initial_pos)
        batch = generator.next()
        for batch_input_code, batch_tensor in batch.items():
            batch[batch_input_code] = batch_tensor.to(DEVICE)
        batch_inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        target_summaries = tokenizer.batch_decode(batch.pop('labels'), skip_special_tokens=True, clean_up_tokenization_spaces=False)

        pred_summaries = trained_model.generate(**batch_inputs, **gen_kwargs)
        pred_summaries = tokenizer.batch_decode(pred_summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if kwargs['store-gen']:
            batch_queries_contexts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for query_context, target_summary, pred_summary in zip(batch_queries_contexts, target_summaries, pred_summaries):
                template['query_context'].append(query_context)
                template['actual_summary'].append(target_summary)
                template['pred_summary'].append(pred_summary)

        scores = get_rouge_score(pred_summaries, target_summaries)
        for score_code, score_val in scores.items():
            if score_code not in avg_scores: avg_scores[score_code] = []
            avg_scores[score_code].append(score_val * 100)
            scores[score_code] = score_val * 100

        running_avg_scores = {score_code: np.average(avg_scores[score_code]) for score_code in avg_scores.keys()}
        p_bar.set_postfix(running_avg_scores)
        p_bar.update(1)

    for score_code, score_list in avg_scores.items():
        avg_scores[score_code] = np.average(score_list)
        
    if not kwargs['store-gen']: return avg_scores
    indices = list(range(1, len(template['query_context']) + 1))
    generations = pd.DataFrame(data=template, index=indices)
    return avg_scores, generations

def _test_model(model, tokenizer, data_loader, model_ckpt_dir, **kwargs):
    tag = kwargs['tag']
    ckpt_path = model_ckpt_dir + '/ckpt_epoch_' + tag + '.pt'
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    model.to(DEVICE)
    model.eval()

    if not kwargs['store-gen']: rouge_scores = _evaluate_trained_model(model, tokenizer, data_loader, **kwargs)
    else: 
        rouge_scores, generations = _evaluate_trained_model(model, tokenizer, data_loader, **kwargs)
        generations.to_csv(model_ckpt_dir + '/better_beam_generations_5.csv', index=False)
    with open(model_ckpt_dir + '/better_beam_test_eval_5.log', 'a') as file:
        file.write('GREEDY (ROUGE-ELI5): ' + str(rouge_scores))

def _run_evaluation(model, tokenizer, data_loader, model_ckpt_dir, **kwargs):
    tags = [str(i + 10) for i in range(1, 11)]
    for tag in tags:
        ckpt_path = model_ckpt_dir + '/ckpt_epoch_' + tag + '.pt'
        model.load_state_dict(torch.load(ckpt_path), strict=False)
        model.to(DEVICE)
        model.eval()
        print('Starting evaluation with model loaded from ckpt# {}'.format(tag))
        rouge_scores = _evaluate_trained_model(model, tokenizer, data_loader, **kwargs)

        with open(model_ckpt_dir + '/{}_eval.log'.format(kwargs['eval-split']), 'a') as file:
            file.write('TAG: {}:\t{}\n'.format(tag, rouge_scores))

if __name__ == '__main__':
    cfg_mgr = ConfigManager()
    kwargs = cfg_mgr.get_kwargs()
    kwargs['eval-mode'] = True

    tools = get_tools(kwargs['model-name'], kwargs['max-pos-embeddings'], None, None, None, None, None, None, **kwargs)
    model = tools['model']
    tokenizer = tools['tokenizer']

    kwargs['pad-token-index'] = tokenizer.pad_token_id
    kwargs['mask-token-index'] = tokenizer.mask_token_id
    kwargs['bos-token-index'] = model.config.bos_token_id
    kwargs['eos-token-index'] = model.config.eos_token_id
    kwargs['decoder-start-token-id'] = model.config.decoder_start_token_id
    kwargs['saved-data'] = False
    kwargs['store-gen'] = True
    kwargs['eval-split'] = 'test'
    kwargs['eval-batch-size'] = 4
    best_tags = {'./saved_model_mle': '5', './saved_model_rl': '19', './saved_model_rl_scheduled_sample': '3',
            './small_saved_model_mle': '8', './small_saved_model_rl_scheduled_sample': '5', 
            './saved_model_rl_scheduled_sample_grad': '4', './saved_model_rl_scheduled_sample_grad_semantic': '5',
            './saved_model_rl_scheduled_sample_grad_bleu': '5', './saved_model_rl_scheduled_sample_grad_f': '5',
            './saved_model_rl_scheduled_sample_grad_pe_sem': '5', './saved_model_rl_scheduled_sample_grad_pe_sem_large': '5',
            './saved_model_rl_scheduled_sample_grad_pe_sem_large_only': '5', './saved_model_rl_scheduled_sample_grad_sbert': '5'}

    data_loader = get_s2s_dataloader(kwargs['base-data-dir'], kwargs['eval-split'], tokenizer, kwargs['decoder-start-token-id'], kwargs['eval-batch-size'], kwargs['saved-data'], **kwargs)

    for model_base_dir in ['./saved_model_rl_scheduled_sample_grad_sbert']: #, './saved_model_rl']:
        kwargs['model-base-dir'] = model_base_dir
        if kwargs['eval-split'] == 'valid': _run_evaluation(model, tokenizer, data_loader, kwargs['model-base-dir'], **kwargs)
        elif kwargs['eval-split'] == 'test':
            kwargs['tag'] = best_tags[model_base_dir]
            _test_model(model, tokenizer, data_loader, kwargs['model-base-dir'], **kwargs)
