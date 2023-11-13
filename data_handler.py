from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random
import torch
import numpy as np
import pickle as pkl
from constants import CAUSAL_LM_TASK, MASKED_LM_TASK, S2S_TASK

class _BaseDataset:
    def __init__(self, base_dir, split, saved):
        if not saved: raw_data_points = self.load_raw_data(base_dir, split)
        if split == 'train':
            if not saved: self.mlm_data, self.causal_lm_data, self.s2s_data = self.process_data(raw_data_points, train=True)
            else:
                with open('multi_task_data/train_mlm_data.pkl', 'rb') as file:
                    self.mlm_data = pkl.load(file)

                with open('multi_task_data/train_clm_data.pkl', 'rb') as file:
                    self.causal_lm_data = pkl.load(file)

                with open('multi_task_data/train_s2s_data.pkl', 'rb') as file:
                    self.s2s_data = pkl.load(file)
        else:
            if not saved: self.eval_data = self.process_data(raw_data_points)
            else:
                with open('multi_task_data/{}_s2s_data.pkl'.format(split), 'rb') as file:
                    self.eval_data = pkl.load(file)

    def load_raw_data(self, base_dir, type):
        source_file_name = base_dir + '/{}.multitask_source'.format(type)
        target_file_name = base_dir + '/{}.multitask_target'.format(type)

        with open(source_file_name, 'r') as file:
            source_data = file.read()
        with open(target_file_name, 'r') as file:
            target_data = file.read()

        source_data = source_data.split('\n')
        target_data = target_data.split('\n')

        return list(zip(source_data, target_data))

    def process_data(self, raw_data_points, train=False):
        if train:
            mlm_data, causal_lm_data, s2s_data = [], [], []
            for source, target in tqdm(raw_data_points, desc='Forming input pipeline'):
                if source[:8] == CAUSAL_LM_TASK: 
                    # Task 1 - Causal LM
                    if len(target) > 0: causal_lm_data.append(target)
                elif source[:8] == MASKED_LM_TASK:
                    # Task 2 - Masked LM
                    if len(source[8:]) > 0: mlm_data.append(source[8:])
                elif source[:8] == S2S_TASK:
                    # Task 3 - S2S
                    query_context = source[8:]
                    if len(query_context) > 0 and len(target) > 0:
                        s2s_data.append((query_context, target))
                elif source == '': pass
                else:
                    print('Error - no obvious task while processing data')

            return mlm_data, causal_lm_data, s2s_data

        else:
            s2s_data = []
            for source, target in tqdm(raw_data_points, desc='Forming input pipeline'):
                if source[:8] == S2S_TASK:
                    # Task 3 - S2S
                    query_context = source[8:]
                    if len(query_context) > 0 and len(target) > 0:
                        s2s_data.append((query_context, target))
                elif source == '': pass 
                else:
                    print('Error - no obvious task while processing data')

            return s2s_data

class _CausalLMDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, **kwargs):
        self.causal_lm_data = base_dataset.causal_lm_data
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __len__(self):
        return len(self.causal_lm_data)

    def __getitem__(self, idx):
        batch = [self.causal_lm_data[idx]]

        tokenized_data = self.tokenizer(batch, padding=True, return_tensors='np')
        tokenized_batch, attention_mask = tokenized_data['input_ids'], tokenized_data['attention_mask']

        tokenized_batch = tokenized_batch[:, :self.kwargs['max-lm-seq-len']]
        attention_mask = attention_mask[:, :self.kwargs['max-lm-seq-len']]

        upto = (tokenized_batch.shape[1] // self.kwargs['lm-aux-seq-len']) * self.kwargs['lm-aux-seq-len']
        tokenized_batch = tokenized_batch[:, :upto]
        attention_mask = attention_mask[:, :upto]

        tokenized_batch = tokenized_batch.reshape(-1, self.kwargs['lm-aux-seq-len'])
        attention_mask = attention_mask.reshape(-1, self.kwargs['lm-aux-seq-len'])

        batch_input, batch_output = tokenized_batch[:, :-1], tokenized_batch[:, 1:]
        return torch.as_tensor(batch_input), torch.as_tensor(batch_output), torch.as_tensor(attention_mask)

class _MaskedLMDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, **kwargs):
        self.mlm_data = base_dataset.mlm_data
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def __len__(self):
        return len(self.mlm_data)

    def __getitem__(self, idx):
        batch_output = [self.mlm_data[idx]]

        batch_input_data, batch_output_data = self.tokenizer(batch_output, padding=True, return_tensors='np'), self.tokenizer(batch_output, padding=True, return_tensors='np')
        batch_input, attention_mask = batch_input_data['input_ids'], batch_input_data['attention_mask']
        batch_output = batch_output_data['input_ids']

        batch_input, batch_output, attention_mask = batch_input[:, :self.kwargs['max-lm-seq-len']], batch_output[:, :self.kwargs['max-lm-seq-len']], attention_mask[:, :self.kwargs['max-lm-seq-len']]
        batch_input = np.array(self._mask_tokens(batch_input))

        upto = (batch_input.shape[1] // self.kwargs['lm-aux-seq-len']) * self.kwargs['lm-aux-seq-len']
        batch_input = batch_input[:, :upto]
        batch_output = batch_output[:, :upto]
        attention_mask = attention_mask[:, :upto]

        batch_input = batch_input.reshape(-1, self.kwargs['lm-aux-seq-len'])
        batch_output = batch_output.reshape(-1, self.kwargs['lm-aux-seq-len'])
        attention_mask = attention_mask.reshape(-1, self.kwargs['lm-aux-seq-len'])

        return torch.as_tensor(batch_input), torch.as_tensor(batch_output), torch.as_tensor(attention_mask)

    def _mask_tokens(self, tokenized_documents):
        processed_docs = []
        for doc in tokenized_documents:
            processed_doc = []
            for token in doc:
                if random.uniform(0, 1) <= self.kwargs['mask-proba'] and token != self.kwargs['pad-token-index']: processed_doc.append(self.kwargs['mask-token-index'])
                else: processed_doc.append(token)
            processed_docs.append(processed_doc)

        return processed_docs

class _QFSDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, decoder_start_token_id, split='train', **kwargs):
        if split == 'train':
            self.s2s_data = base_dataset.s2s_data
        else:
            self.s2s_data = base_dataset.eval_data
        self.split = split
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.decoder_start_token_id = decoder_start_token_id

    def __len__(self):
        return len(self.s2s_data)

    def __getitem__(self, idx):
        batch = [self.s2s_data[idx]]

        source_queries_contexts, target = tuple(zip(*batch))
        source_queries_contexts = list(source_queries_contexts)
        target = list(target)

        tokenized_source_queries_contexts_data = self.tokenizer(source_queries_contexts, padding=True, return_tensors='np')
        tokenized_source_queries_contexts, encoder_attention_mask = tokenized_source_queries_contexts_data['input_ids'], tokenized_source_queries_contexts_data['attention_mask']

        tokenized_target_data = self.tokenizer(target, padding=True, return_tensors='np')
        tokenized_target, target_attention_mask = tokenized_target_data['input_ids'], tokenized_target_data['attention_mask']
        
        target_input, target_output = tokenized_target[:, :-1], tokenized_target[:, 1:]
        target_attention_mask = target_attention_mask[:, :-1]
        target_input[:, 0] = self.decoder_start_token_id

        tokenized_source_queries_contexts = tokenized_source_queries_contexts[:, :self.kwargs['max-pos-embeddings']]
        target_input = target_input[:, :self.kwargs['max-pos-embeddings']]
        target_output = target_output[:, :self.kwargs['max-pos-embeddings']]
        encoder_attention_mask = encoder_attention_mask[:, :self.kwargs['max-pos-embeddings']]
        target_attention_mask = target_attention_mask[:, :self.kwargs['max-pos-embeddings']]

        return torch.as_tensor(tokenized_source_queries_contexts).squeeze(), \
                torch.as_tensor(target_input).squeeze(), torch.as_tensor(target_output).squeeze(), \
                torch.as_tensor(encoder_attention_mask).squeeze(), torch.as_tensor(target_attention_mask).squeeze()

def lm_collate(batch):
    inputs, outputs, attention_masks = tuple(zip(*batch)) # inputs and outputs aren't ragged
    input_batch = torch.cat(inputs)
    output_batch = torch.cat(outputs)
    batch_attention_mask = torch.cat(attention_masks)

    return {
        'input_ids': input_batch,
        'attention_mask': batch_attention_mask,
        'labels': output_batch
    }

def s2s_collate(batch, pad_token_index):
    encoder_inputs, decoder_inputs, decoder_outputs, enc_attention_masks, dec_attention_masks = tuple(zip(*batch)) # inputs and outputs are ragged!

    encoder_input_batch = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_index)
    decoder_input_batch = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_token_index)
    decoder_output_batch = pad_sequence(decoder_outputs, batch_first=True, padding_value=pad_token_index)
    enc_attention_mask_batch = pad_sequence(enc_attention_masks, batch_first=True, padding_value=0) # where ever padded, no attention, which is 0
    dec_attention_mask_batch = pad_sequence(dec_attention_masks, batch_first=True, padding_value=0) # where ever padded, no attention, which is 0
    
    assert dec_attention_mask_batch.size() == decoder_input_batch.size()

    return {
        'input_ids': encoder_input_batch,
        'attention_mask': enc_attention_mask_batch,
        'decoder_input_ids': decoder_input_batch,
        # 'decoder_attention_mask': dec_attention_mask_batch,
        'labels': decoder_output_batch
    }

SPLIT_TRAIN = 'train'
SPLIT_VALID = 'valid'
SPLIT_TEST = 'test'

SHUFFLE_DATA = True
NUM_WORKERS = 0
DROP_LAST = False
PRE_FETCH_FACTOR = 4 * NUM_WORKERS

base_datasets = {
    SPLIT_TRAIN: None,
    SPLIT_VALID: None,
    SPLIT_TEST: None 
}

def get_base_dataset(base_dir, split, saved):
    if base_datasets[split] is None:
        base_datasets[split] = _BaseDataset(base_dir, split, saved)
    return base_datasets[split]

def get_masked_lm_dataloader(base_dir, split, tokenizer, batch_size, saved=False, **kwargs):
    base_dataset = get_base_dataset(base_dir, split, saved)
    masked_lm_dataset = _MaskedLMDataset(base_dataset, tokenizer, **kwargs)
    data_loader = DataLoader(
        dataset=masked_lm_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE_DATA,
        num_workers=NUM_WORKERS,
        collate_fn=lm_collate,
        drop_last=DROP_LAST,
        # prefetch_factor=PRE_FETCH_FACTOR
    )

    return data_loader

def get_causal_lm_dataloader(base_dir, split, tokenizer, batch_size, saved=False, **kwargs):
    base_dataset = get_base_dataset(base_dir, split, saved)
    causal_lm_dataset = _CausalLMDataset(base_dataset, tokenizer, **kwargs)
    data_loader = DataLoader(
        dataset=causal_lm_dataset,
        batch_size=batch_size,
        shuffle=SHUFFLE_DATA,
        num_workers=NUM_WORKERS,
        collate_fn=lm_collate,
        drop_last=DROP_LAST,
        # prefetch_factor=PRE_FETCH_FACTOR
    )

    return data_loader

def get_s2s_dataloader(base_dir, split, tokenizer, decoder_start_token_id, batch_size, saved=False, **kwargs):
    base_dataset = get_base_dataset(base_dir, split, saved)
    s2s_dataset = _QFSDataset(base_dataset, tokenizer, decoder_start_token_id, split, **kwargs)

    shuffle = (SHUFFLE_DATA and split=='train')

    data_loader = DataLoader(
        dataset=s2s_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: s2s_collate(batch, tokenizer.pad_token_id),
        drop_last=DROP_LAST,
        # prefetch_factor=PRE_FETCH_FACTOR
    )

    return data_loader