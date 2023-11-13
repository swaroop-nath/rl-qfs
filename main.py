from model_code import get_tools
from data_handler import get_causal_lm_dataloader, get_masked_lm_dataloader, get_s2s_dataloader
from allennlp.training.learning_rate_schedulers.noam import NoamLR
from train_causal_lm import run_batch as run_causal_batch
from train_masked_lm import run_batch as run_masked_batch
from train_qfs_mle import run_batch as run_qfs_mle_batch
from train_qfs_pg import run_batch as run_qfs_rl_batch
from tqdm import tqdm
import numpy as np
import torch
from config import ConfigManager
from constants import DEVICE

CAUSAL_LM_TRAIN_TASK = 'causal_lm'
MASKED_LM_TRAIN_TASK = 'masked_lm'
QFS_MLE_TRAIN_TASK = 'qfs_mle'
QFS_RL_TRAIN_TASK = 'qfs_rl'

progress_log_file = open('progress/progress_rl_grad.log', 'w')
item_logger_file = 'progress/loss_items_rl_grad.log'
initial_pos = progress_log_file.tell()
log_every = 16000

NUM_CUDA_DEVICES = torch.cuda.device_count()

CKPT_PATH_TEMPLATE = 'ckpt_epoch_{}.pt'

def train_one_batch(ds_engine, optimizer, scheduler, apply_grad, task_runners, batches, tokenizer, **kwargs):
    loss_items = {}
    total_loss = 0
    for task_code, batch_runner in task_runners.items(): # task_runners = {task_code: run_batch fn pointer}
        batch = batches[task_code]

        for batch_key, batch_tensor in batch.items():
            batch[batch_key] = batch_tensor.to(DEVICE)

        outputs = batch_runner(ds_engine, batch, tokenizer=tokenizer, **kwargs)
        loss = outputs['loss']

        if task_code == QFS_MLE_TRAIN_TASK and QFS_RL_TRAIN_TASK in task_runners:
            logits = outputs['logits'].detach().clone()
            mle_gen = torch.argmax(logits, dim=-1)
            kwargs['mle-gen'] = mle_gen

        if task_code == CAUSAL_LM_TRAIN_TASK: weight = kwargs['clm-weight']
        elif task_code == MASKED_LM_TRAIN_TASK: weight = kwargs['mlm-weight']
        elif task_code == QFS_MLE_TRAIN_TASK: weight = kwargs['qfs-mle-weight']
        elif task_code == QFS_RL_TRAIN_TASK: weight = kwargs['qfs-rl-weight']

        total_loss += weight * loss

        if task_code != QFS_RL_TRAIN_TASK: loss_items[task_code] = loss.clone().detach().item()
        else:
            loss_items[task_code + '_loss'] = loss.clone().detach().item()
            loss_items[task_code + '_marginal_reward'] = outputs['marginal_reward'].clone().detach().item()
            loss_items[task_code + '_reward'] = outputs['reward'].clone().detach().item()

    ds_engine.backward(total_loss)
    ds_engine.step() # Automatically handles grad accumulation

    if apply_grad:
        if scheduler is not None and isinstance(scheduler, NoamLR): scheduler.step_batch()
        prof_force_ratio = 0.65 - 0.5 / (1 + np.exp(-0.02 * (kwargs['prof-force-step-ctr'] - 10)))
        kwargs['prof-force-proba'] = min(prof_force_ratio, kwargs['prof-force-proba-max'])
        kwargs['prof-force-step-ctr'] += 1

    return loss_items

def train_one_epoch(ds_engine, optimizer, scheduler, task_loaders, save_dir, epoch, tokenizer, **kwargs):
    with open(item_logger_file, 'a') as file:
        file.write('\n' + '='*50 + '\nStart of Epoch {}\n'.format(epoch) + '='*50 + '\n')
    num_batches = len(list(task_loaders.values())[0])
    task_runners = {}
    running_window_len = 1000
    running_losses = {}
    avg_losses = {}
    insert_idx = 0
    for task_code in task_loaders.keys():
        if task_code == CAUSAL_LM_TRAIN_TASK: task_runners[task_code] = run_causal_batch
        elif task_code == MASKED_LM_TRAIN_TASK: task_runners[task_code] = run_masked_batch
        elif task_code == QFS_MLE_TRAIN_TASK: task_runners[task_code] = run_qfs_mle_batch
        elif task_code == QFS_RL_TRAIN_TASK: task_runners[task_code] = run_qfs_rl_batch

    p_bar = tqdm(total=num_batches, desc='Training', leave=True, file=progress_log_file)
    for step in range(num_batches):
        progress_log_file.seek(initial_pos)
        apply_grad = ((step + 1) % kwargs['grad-acc-every'] == 0) or (step == num_batches - 1)
        batches = {}
        for task_code, data_loader in task_loaders.items(): # task_loaders = {task_code: data_loader}
            batches[task_code] = next(iter(data_loader)) # Faulty but ok in the long run
        loss_items = train_one_batch(ds_engine, optimizer, scheduler, apply_grad, task_runners, batches, tokenizer, **kwargs)
        
        for task_code, loss_item in loss_items.items():
            if task_code not in running_losses:
                running_losses[task_code] = [0] * running_window_len
            if task_code not in avg_losses:
                avg_losses[task_code] = 0
            running_losses[task_code][insert_idx] = loss_item
            avg_losses[task_code] += loss_item / log_every
        insert_idx = (insert_idx + 1) % running_window_len
        p_bar.update(1)

        if step < running_window_len: p_bar.set_postfix(loss_items)
        else:
            running_losses_avgs = {'running ' + task_code: np.average(running_loss) for task_code, running_loss in running_losses.items()}
            p_bar.set_postfix(running_losses_avgs)

        if (step + 1) % log_every == 0:
            with open(item_logger_file, 'a') as file:
                file.write(str(avg_losses) + '\n')
                avg_losses = {task_code: 0 for task_code in avg_losses.keys()}
    p_bar.close()

    save_item = {'prof-force-step-ctr': kwargs['prof-force-step-ctr']}
    if scheduler is not None:
        save_item['scheduler-state-dict'] = scheduler.state_dict()
    ds_engine.save_checkpoint(save_dir, epoch, client_state=save_item)

    return ds_engine, optimizer, scheduler

if __name__ == '__main__':
    cfg_mgr = ConfigManager()
    kwargs = cfg_mgr.get_kwargs()
    if kwargs.get('schedule-policy') is None: kwargs['schedule-policy'] = None
    kwargs['eval-mode'] = False
    batch_per_gpu = kwargs['batch-size'] / NUM_CUDA_DEVICES

    tools = get_tools(kwargs['model-name'], kwargs['max-pos-embeddings'], kwargs['optim-name'], kwargs['learning-rate'], kwargs['schedule-policy'], kwargs['ds-config-file-path'], batch_per_gpu, kwargs['grad-acc-every'], **kwargs)

    ds_engine = tools['ds-engine']
    tokenizer = tools['tokenizer']
    optimizer = tools['optimizer']
    scheduler = tools['scheduler']
    ds_config_hf = tools['ds-config-hf']

    if kwargs['start-epoch'] > 1:
        _, client_sd = ds_engine.load_checkpoint(kwargs['save-dir'], kwargs['start-epoch'] - 1)
        if 'prof-force-step-ctr' in client_sd: kwargs['prof-force-step-ctr'] = client_sd['prof-force-step-ctr']
        else: kwargs['prof-force-step-ctr'] = 0
    else: kwargs['prof-force-step-ctr'] = 0  

    prof_force_ratio = 0.65 - 0.5 / (1 + np.exp(-0.02 * (kwargs['prof-force-step-ctr'] - 10)))
    kwargs['prof-force-proba'] = min(prof_force_ratio, kwargs['prof-force-proba-max'])
        
    kwargs['pad-token-index'] = tokenizer.pad_token_id
    kwargs['mask-token-index'] = tokenizer.mask_token_id
    kwargs['decoder-start-token-id'] = ds_engine.module.config.decoder_start_token_id

    # tasks = [CAUSAL_LM_TRAIN_TASK, MASKED_LM_TRAIN_TASK, QFS_MLE_TRAIN_TASK] # No support for Causal LM and Masked LM yet
    tasks = [QFS_MLE_TRAIN_TASK]
    task_loaders = {}
    if not kwargs['mle-qfs']: 
        tasks.append(QFS_RL_TRAIN_TASK)
        kwargs['do-qfs-rl'] = 1
    else:
        kwargs['do-qfs-rl'] = 0

    # Setting weights
    for task_code in tasks:
        total_weight = kwargs['clm-weight'] * kwargs['do-clm'] + \
                        kwargs['mlm-weight'] * kwargs['do-mlm'] + \
                        kwargs['qfs-mle-weight'] * kwargs['do-qfs-mle'] + \
                        kwargs['qfs-rl-weight'] * kwargs['do-qfs-rl']

        if task_code == CAUSAL_LM_TRAIN_TASK: kwargs['clm-weight'] *= kwargs['do-clm'] / total_weight
        elif task_code == MASKED_LM_TRAIN_TASK: kwargs['mlm-weight'] *= kwargs['do-mlm'] / total_weight
        elif task_code == QFS_MLE_TRAIN_TASK: kwargs['qfs-mle-weight'] *= kwargs['do-qfs-mle'] / total_weight
        elif task_code == QFS_RL_TRAIN_TASK: kwargs['qfs-rl-weight'] *= kwargs['do-qfs-rl'] / total_weight

    for task_code in tasks:
        if task_code == CAUSAL_LM_TRAIN_TASK: task_loaders[task_code] = get_causal_lm_dataloader(kwargs['base-data-dir'], 'train', tokenizer, kwargs['batch-size'], kwargs['saved-data'], **kwargs)
        elif task_code == MASKED_LM_TRAIN_TASK: task_loaders[task_code] = get_masked_lm_dataloader(kwargs['base-data-dir'], 'train', tokenizer, kwargs['batch-size'], kwargs['saved-data'], **kwargs)
        elif task_code == QFS_MLE_TRAIN_TASK: task_loaders[task_code] = get_s2s_dataloader(kwargs['base-data-dir'], 'train', tokenizer, kwargs['decoder-start-token-id'], kwargs['batch-size'], kwargs['saved-data'], **kwargs)
        elif task_code == QFS_RL_TRAIN_TASK: task_loaders[task_code] = get_s2s_dataloader(kwargs['base-data-dir'], 'train', tokenizer, kwargs['decoder-start-token-id'], kwargs['batch-size'], kwargs['saved-data'], **kwargs)

    for epoch in range(kwargs['start-epoch'], kwargs['num-epoch'] + 1):
        ds_engine, optimizer, scheduler = train_one_epoch(ds_engine, optimizer, scheduler, task_loaders, kwargs['save-dir'], epoch, tokenizer, **kwargs)