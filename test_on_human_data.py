import json
from utils import get_rouge_score
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import ConfigManager
from model_code import get_tools
import torch

DEVICE = 'cuda:0'

def generate_summary(model, query, support_doc, summary, tokenizer):
    greedy_kwargs = {'min_length': 64, 'max_length': 256}
    beam_kwargs = {'min_length': 64, 'max_length': 256, 'num_beams': 15, 'no_repeat_ngram_size': 2} # Better results
    # beam_kwargs = {'min_length': 64, 'max_length': 256, 'num_beams': 5, 'no_repeat_ngram_size': 3} # Usually beam works better, much better
    top_p_top_k_kwargs = {'min_length': 64, 'max_length': 256, 'do_sample': True, 'temperature': 1, 'top_p': 0.8, 'no_repeat_ngram_size': 3, 'num_beams': 5, 'top_k': 100}

    gen_kwargs = beam_kwargs

    query_support_doc = query + " " + support_doc
    tokenized_query_summary = tokenizer(query_support_doc, return_tensors='pt')
    batch_input = {'input_ids': tokenized_query_summary['input_ids'][:, :1568].to(DEVICE), \
                    'attention_mask': tokenized_query_summary['attention_mask'][:, :1568].to(DEVICE)}
    pred_summaries = model.generate(**batch_input, **gen_kwargs)

    pred_summary = tokenizer.batch_decode(pred_summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    scores = get_rouge_score([pred_summary], [summary])

    return pred_summary, scores

def evaluate_model(model, save_dir, dataset_paths, tokenizer, **kwargs):
    with open(dataset_paths['qfs-path'], 'r') as file:
        qfs_lines = file.readlines()
    with open(dataset_paths['support-docs-path'], 'r') as file:
        support_docs_lines = file.readlines()

    support_docs = {}
    list_of_doc_ids = []
    for doc in support_docs_lines:
        doc = json.loads(doc)
        support_docs[doc['doc_id']] = doc['document']
        list_of_doc_ids.append(doc['doc_id'])

    avg_scores = {}
    template = {'query': [], 'context': [], 'actual_summary': [], 'pred_summary': []}
    p_bar = tqdm(total=len(qfs_lines), desc='Evaluating')
    for line in qfs_lines:
        data = json.loads(line)
        if "summary" not in data: continue
        query = data['query']
        summary = data['summary']
        if not kwargs['random-doc']: support_doc = support_docs[data['support_doc']]
        else: 
            random_doc_id_choice_list = list(set(list_of_doc_ids) - set([data['support_doc']]))
            random_doc_id = np.random.choice(random_doc_id_choice_list, 1)[0]
            support_doc = support_docs[random_doc_id]

        pred_summary, scores = generate_summary(model, query, support_doc, summary, tokenizer)
        for score_code, score_val in scores.items():
            if score_code not in avg_scores: avg_scores[score_code] = []
            avg_scores[score_code].append(score_val * 100)
            scores[score_code] = score_val * 100

        template['query'].append(query)
        template['context'].append(support_doc)
        template['actual_summary'].append(summary)
        template['pred_summary'].append(pred_summary)
        running_avg_scores = {score_code: np.average(avg_scores[score_code]) for score_code in avg_scores.keys()}
        p_bar.set_postfix(running_avg_scores)
        p_bar.update(1)
    p_bar.close()

    for score_code, score_list in avg_scores.items():
        avg_scores[score_code] = np.average(score_list)

    print(avg_scores)
    indices = list(range(1, len(template['query']) + 1))
    generations = pd.DataFrame(data=template, index=indices)
    # if not kwargs['random-doc']: 
    #     generations.to_csv(save_dir + '/better_beam_generations_human_eval.csv', index=False)
    #     with open(save_dir + '/better_beam_eval_human.log', 'w') as file:
    #         file.write('Scores: ' + str(avg_scores))
    # else:
    #     generations.to_csv(save_dir + '/better_beam_generations_human_eval_random.csv', index=False)
    #     with open(save_dir + '/better_beam_eval_human_random.log', 'w') as file:
    #         file.write('Scores: ' + str(avg_scores))

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
    kwargs['random-doc'] = False

    model_list = {'saved_model_mle': '5',
        'saved_model_rl_scheduled_sample_grad': '5', 'saved_model_rl_scheduled_sample_grad_semantic': '5', 
        'saved_model_rl_scheduled_sample_grad_bleu': '5',
        'saved_model_rl_scheduled_sample_grad_pe_sem_large': '5', 'saved_model_rl_scheduled_sample_grad_pe_sem_large_only': '5'}

    dataset_paths = {'qfs-path': 'annotation/dataset.jsonl', 'support-docs-path': 'annotation/support_docs.jsonl'}

    for model_ckpt_dir, tag in {'./saved_model_rl_scheduled_sample_grad_sbert': '5'}.items():
        print(model_ckpt_dir + '\n')
        ckpt_path = model_ckpt_dir + '/ckpt_epoch_' + tag + '.pt'
        model.load_state_dict(torch.load(ckpt_path), strict=False)
        model.to(DEVICE)
        model.eval()

        evaluate_model(model, model_ckpt_dir, dataset_paths, tokenizer, **kwargs)