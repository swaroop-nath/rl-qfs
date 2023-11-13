from data_handler import get_causal_lm_dataloader, get_masked_lm_dataloader, get_s2s_dataloader
from config import ConfigManager
from transformers import BartTokenizerFast
from tqdm import tqdm
from model_code import get_tools

def _get_kwargs():
    cfg = ConfigManager()
    return cfg.get_kwargs()

def _test_dataloaders_lengths(data_loader, dataset_desc):
    print(dataset_desc + ': ' + str(len(data_loader)))

def _test_seq_len(data_loader, split):
    input_fp = open('input_seq_len_{}.log'.format(split), 'a')
    output_fp = open('output_seq_len_{}.log'.format(split), 'a')
    for batch in tqdm(data_loader, desc='Running through S2S data'):
        input_seq_len = batch['input_ids'].size(1)
        output_seq_len = batch['labels'].size(1)
        input_fp.write(str(input_seq_len) + '\n')
        output_fp.write(str(output_seq_len) + '\n')
    input_fp.close()
    output_fp.close()

def _test_out_labels(data_loader, vocab_size):
    for batch in tqdm(data_loader, desc='Running through S2S data'):
        is_input_greater = (batch['input_ids'] >= vocab_size).any().item()
        is_output_greater = (batch['labels'] >= vocab_size).any().item()
        if is_input_greater or is_output_greater:
            raise RuntimeError

def _test_dataloader_iters(data_loader, desc):
    MAX = 1e8
    ctr = 0
    for batch in tqdm(data_loader, desc=desc):
        ctr += 1
        if ctr == MAX: break

def _test_dataloaders(base_dir, tokenizer, batch_size, **kwargs):
    test_suite = {'lengths': False, 'iters': False, 'max-len': False, 'out-labels': True}
    for split in ['valid', 'train', 'test']:
        if split == 'train':
            causal_lm_dataloader = get_causal_lm_dataloader(base_dir, split, tokenizer, batch_size, saved=kwargs['saved-data'], **kwargs)
            masked_lm_dataloader = get_masked_lm_dataloader(base_dir, split, tokenizer, batch_size, saved=kwargs['saved-data'], **kwargs)

        s2s_dataloader = get_s2s_dataloader(base_dir, split, tokenizer, tokenizer.eos_token_id, batch_size, saved=kwargs['saved-data'], **kwargs)

        if split == 'train':
            if test_suite['iters']:
                _test_dataloader_iters(causal_lm_dataloader, desc='Running through causal lm dataset')
                _test_dataloader_iters(masked_lm_dataloader, desc='Running through masked lm dataset')
            if test_suite['lengths']:
                _test_dataloaders_lengths(causal_lm_dataloader, dataset_desc='Causal LM Dataset')
                _test_dataloaders_lengths(masked_lm_dataloader, dataset_desc='Masked LM Dataset')

        if test_suite['iters']:
            _test_dataloader_iters(s2s_dataloader, desc='Running through s2s dataset of split: {}'.format(split))
        if test_suite['lengths']:
            _test_dataloaders_lengths(s2s_dataloader, dataset_desc='S2S Dataset')
        if test_suite['max-len']:
            s2s_dataloader_aux = get_s2s_dataloader(base_dir, split, tokenizer, tokenizer.eos_token_id, 1, saved=kwargs['saved-data'], **kwargs)
            _test_seq_len(s2s_dataloader_aux, split)
        if test_suite['out-labels']:
            _test_out_labels(s2s_dataloader, kwargs['vocab-size'])

def _test_tools(**kwargs):
    model_name = 'facebook/bart-large'
    optim_name = 'adam'
    learning_rate = 1e-5
    schedule_policy = None
    ds_config_file_path = './ds_config.json'

    batch_per_gpu = 1
    grad_acc_steps = 8

    tools = get_tools(model_name, kwargs['max-pos-embeddings'], optim_name, learning_rate, \
        schedule_policy, ds_config_file_path, batch_per_gpu, grad_acc_steps, **kwargs) 

    assert tools is not None

    ds_engine = tools['ds-engine']
    optimizer = tools['optimizer']
    dschf = tools['ds-config-hf']
    assert ds_engine.module is not None
    assert optimizer is not None
    assert dschf is not None

    assert ds_engine.load_checkpoint(kwargs['save-dir'], kwargs['start-epoch'] - 1) is not None

if __name__ == '__main__':
    kwargs = _get_kwargs()

    test_suite = {'tools': True,
                'data_loader': False}

    # Testing dataloaders
    if test_suite['data_loader']:
        base_dir = 'multi_task_data'
        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
        kwargs['pad-token-index'] = tokenizer.pad_token_id
        kwargs['mask-token-index'] = tokenizer.mask_token_id
        kwargs['vocab-size'] = tokenizer.vocab_size
        kwargs['saved-data'] = True
        batch_size = 4
        _test_dataloaders(base_dir, tokenizer, batch_size, **kwargs)

    # Testing tools
    if test_suite['tools']:
        kwargs['save-dir'] = './saved_model_rl'
        kwargs['start-epoch'] = 21
        kwargs['eval-mode'] = False
        _test_tools(**kwargs)