import argparse
import time
import random
import numpy as np
import torch
import torch.autograd.profiler as profiler
import json
from pprint import pprint as pp
from tasks.data_loaders.data_utils import get_train_data_loader, get_eval_data_loader
from modules.utils import gpt_loss_func
from modules.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module
from pathlib import Path
import time
import shutil

from transformers import AutoConfig, PretrainedConfig, TrainerCallback, TrainerControl
import datasets

# import wandb
from utils.dist_args_utils import *
from utils.dist_checkpoint_utils import *
from comm.comm_utils import *
import compress.flag
import subprocess

class ProgressCallback(TrainerCallback):
    def __init__(self, log_file_path="/app/mnt/progress.log"):
        self.log_file_path = log_file_path
        self.log_file = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Open the log file at the start of training
        try:
            self.log_file = open(self.log_file_path, "a")
        except Exception as e:
            print(f"Error opening log file: {e}")
            exit(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if True:  # Only log for the main process in distributed training
        # if state.is_local_process_zero:  # Only log for the main process in distributed training
            log_message = f"Step: {state.global_step}, Logs: {logs}\n"
            try:
                self.log_file.write(log_message)
                self.log_file.flush()  # Ensure the log is written immediately
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        # Close the log file at the end of training
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")

def test_loop(args, pipe, device, test_data_loader):

    if test_data_loader is None:
        return

    print('testing starts.....')

    pipe.model.eval()

    if get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:

        def _lm_pred_func(x, y):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits = x[:, :-1, :].contiguous().float()
            labels = y[:, 1:].contiguous()
            loss = loss_fct(logits.transpose(-1, -2), labels).mean(1).detach().cpu()
            return loss

        loss_list = []
        for i, data in enumerate(test_data_loader):

            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break

            input_ids = data['input_ids'].to(device)
            labels = input_ids.clone()
            pipe.infer_iter(input_ids, labels, output_=loss_list, pred_func=_lm_pred_func)

        loss = torch.tensor(loss_list).mean()
        ppls = torch.exp(loss)
        metric = {"valid.perplexity": ppls.item(), "valid.loss": loss.item()}

        print(metric)
        # wandb.log(
        #     metric,
        #     step=pipe.global_step,
        # )

    else:
        for i, data in enumerate(test_data_loader):

            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break

            input_ids = data['input_ids'].to(device)
            labels = input_ids.clone()
            current_iter_time = pipe.infer_iter(input_ids, labels)

    pipe.model.train()



def train_loop(args, pipe, device, train_data_loader, test_data_loader,
               progress: ProgressCallback, control : TrainerControl):

    print('training starts......')
    progress.on_train_begin(args=None, state=pipe, control=control)

    pipe.model.train() # Flag .training to True to enable Dropout

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        # dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    pp_comm = get_pipeline_parallel_comm()

    stop_flag = torch.zeros(1, dtype=torch.int64).to(device)

    input_ids = torch.zeros(
        [args.batch_size, args.seq_length],
        dtype=torch.int64
    ).to(device)

    do_sync_before_save = (args.dp_mode in ['local'] and use_dp)

    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:

        for i, data in enumerate(train_data_loader):
            #if i < pipe.global_step:
                #print(i)
                #continue

            if use_dp:
                get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)

            if stop_flag.item() == 1:
                break

            input_ids_global = data['input_ids'].to(torch.int64).to(device)

            input_ids_list = input_ids_global.chunk(dp_size)

            if use_dp:
                for j in range(1, dp_size):
                    get_data_parallel_comm().send(
                        input_ids_list[j], j,
                    )

            input_ids = input_ids_list[0]

            pp_comm.broadcast(input_ids, 0)

            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)

            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)

            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()

            if pipe.global_step >= args.total_steps:
                stop_flag.data[:] = 1

    elif get_pipeline_parallel_rank() == 0:

        while True:

            get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break

            get_data_parallel_comm().recv(
                input_ids, 0,
            )
            pp_comm.broadcast(input_ids, 0)

            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)

            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)

            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()


    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:

        while True:

            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break

            pp_comm.broadcast(input_ids, 0)
            labels = input_ids.clone()
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func) # lm loss func

            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)

            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                    pipe.save_on_disk(args.checkpoint_path)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
    else:
        while True:
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
            pp_comm.broadcast(input_ids, 0)
            compress.flag.FLAG_DISABLE_COMPRESSION = (pipe.global_step < args.train_warmup_steps)
            current_iter_time = pipe.sgd_iter(None, None)

            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)

            if pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()

    progress.on_train_end(args=None, state=pipe, control=control)


def load_args_from_json(filename="config.json"):
    with open(filename, "r") as f:
        config = json.load(f)
    return config

def get_only_file(folder):
    files = [f.name for f in Path(folder).iterdir() if f.is_file()]
    return files[0] if len(files) == 1 else None

def load_default_config():
    default_config_file = "/app/Cocktail/config.json"
    config = load_args_from_json(default_config_file)
    parser = argparse.ArgumentParser(description='Default-Gpipe-GPT')
    args = parser.parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    return args

def parse_arguments():
    parser = argparse.ArgumentParser(description='Gpipe-GPT')
    parser.add_argument("--data_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path of the pre-trained model.")
    parser.add_argument("--config_path", type=str, default="/app/mnt/config.json", help="Path to the config.json file.")
    parser.add_argument("--output_dir", type=str, default="/app/mnt/output_model", help="Directory to save the fine-tuned model.")
    input_args = parser.parse_args()
    try:
        args = load_default_config()
        config = load_args_from_json(input_args.config_path)
        # Override default argparse values with those from JSON
        for key, value in config.items():
            setattr(args, key, value)

        # output model path, which is different from ckp path
        args.output_dir = input_args.output_dir
        args.model_name = input_args.model_path
        args.tokenizer_name = input_args.model_path
        args.data_path = input_args.data_path

        # create checkpoint path
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

        task_name = config["task_name"]
        # check if input_args.data_path is a directory
        if os.path.isdir(input_args.data_path):
            data_file = get_only_file(args.data_path)
            args.data_path = input_args.data_path + "/" + data_file
        else:
            args.data_path = input_args.data_path

        if not os.path.exists(input_args.output_dir):
            os.makedirs(input_args.output_dir)
        return args
    except Exception as e:
        print("Error parsing arguments:", e)

def finetune(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Start training...")
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    init_communicators(args)

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1

    if args.model_type != 'h3':
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        # H3 does not have AutoConfig
        config = PretrainedConfig.from_dict({
            'n_layer': args.num_layers,
            'd_model': args.embedding_dim,
            'd_inner': args.embedding_dim * 4,
            'vocab_size': 50257,
            'attn_cfg': dict(num_heads = 12), # HARD CODED FOR 125M
            'attn_layer_idx': [1, 8], # HARD CODED FOR 125M
            'ssm_cfg': dict(mode='diag', measure='diag-lin'),
            'pad_vocab_size_multiple': 8,
            'max_position_embeddings': 0,
            'resid_dropout': 0.0,
            'embed_dropout': 0.1,
            'layer_norm_epsilon': 1e-5,
            'fused_mlp': True,
            'fused_dropout_add_ln': True,
            'residual_in_fp32': True
        })
        print(config)

    progress = ProgressCallback()
    control = TrainerControl()

    # num layer globally
    if hasattr(config, 'num_hidden_layers'):
        args.max_layers = config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        args.max_layers = config.num_layers
    else:
        args.max_layers = config.n_layer

    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    # config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", config.vocab_size)

    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:
        train_data_loader = get_train_data_loader(args, tokenizer)
    else:
        train_data_loader = None

    if args.evaluation_data is not None and dp_rank == 0:
        test_data_loader = get_eval_data_loader(args, tokenizer)
    else:
        test_data_loader = None

    if args.total_steps is None:
        args.total_steps = len(train_data_loader)

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

    pipe = get_pp_module(args, config, device, use_dp, progress, control)

    if args.load_checkpoint:
        load_checkpoint(pipe, args)

    if args.fp16:
        pipe.optimizer.reload_model_params()

    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader, progress, control)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader, progress, control)
            except Exception as e:
                raise e
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader, progress, control)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    print(get_pipeline_parallel_rank(), 'finished.')

def pick_checkpoint(args):
    ckp_path = args.checkpoint_path
    out_path = args.output_dir
    # get the last checkpoint
    last_ckp = args.checkpoint_path + "/latest"
    last_ckp_path = None
    with open(last_ckp, "r") as f:
        last_ckp = f.readline().strip()
        last_ckp_path = args.checkpoint_path + f"/checkpoint_{last_ckp}"

    if not os.path.exists(last_ckp_path):
        raise FileNotFoundError(f"Checkpoint {last_ckp_path} not found.")

    # copy the last checkpoint to the output directory
    out_ckp = out_path 
    source_dir = last_ckp_path
    destination_dir = out_ckp + "/checkpoint"
    try:
        shutil.copytree(source_dir, destination_dir)
        print(f"Copied directory {source_dir} to {destination_dir}")
    except FileNotFoundError:
        print(f"Error: The directory {source_dir} does not exist.")
        raise Error(f"Unexpected error: {e}")
    except FileExistsError:
        print(f"Error: The destination {destination_dir} already exists.")
        raise Error(f"Unexpected error: {e}")
    except PermissionError:
        print(f"Error: Permission denied when accessing {source_dir} or {destination_dir}.")
        raise Error(f"Unexpected error: {e}")
    except Exception as e:
        raise Error(f"Unexpected error: {e}")

def main():
    try:
        args = parse_arguments()
        print(args)
    except Exception as e:
        print("Error parsing arguments:", e)
        raise e

    try:
        print("Start finetuning...")
        finetune(args)
        print("Pick the latest checkpoint as output model...")
        pick_checkpoint(args)
    except Exception as e:
        print("Error during training:", e)
        raise e

def retry(func, max_attempts=3, delay=1, *args, **kwargs):
    attempts = 0
    while attempts < max_attempts:
        try:
            return func(*args, **kwargs)  # Try executing the function
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed: {e}")
            if attempts >= max_attempts:
                raise  # Raise the exception if max attempts reached
            time.sleep(delay)  # Optional delay before retrying

if __name__ == '__main__':
    retry(main)
