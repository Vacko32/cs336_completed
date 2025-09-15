from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import BPETrainer
import cs336_basics.transformer.modules as customm
from cs336_basics.transformer.modules import TransformerLM, AdamW, learning_rate_schedule, data_get_batch
import pickle as pkl
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch 
import wandb
from tqdm.auto import tqdm
from torch.profiler import profile, ProfilerActivity, record_function
import sys
from torch.nn.utils import clip_grad_norm_ # more optimal then mine 
import argparse 


def gradient_clipping(params, max_norm = 1.0):
    return clip_grad_norm_(params, max_norm) 


def training_loop(config : dict, run):
    id = config["id"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np_set = np.load(config["src"], mmap_mode='r') # lazy memory mode if file is too big
    len_dataset = np_set.shape[0]
    
    total_steps = int(config["num_training_epochs"] * len_dataset) // (config["batch_size"] * config["context_length"])
    # construct our transformer 


    transformer = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["num_layers"],
        config["d_model"],
        config["num_heads"],
        config["d_ff"],
        config["theta"],
        device,
        config["dtype"],
        config["use_rope"],
        config["use_swiglu"],
    )
    transformer = transformer.to(device)

    beta_tuple = (config["beta1"], config["beta2"])
    optimizer = AdamW(
        transformer.parameters(),
        lr = config["learning_rate"],
        betas=beta_tuple,
        weight_decay=config["weight_decay"],
    )

    """
    Loading checkpoint space 
    """
    # profiling 5 steps only 
    for step in tqdm(range(1, total_steps + 1), unit="step"):

            optimizer.zero_grad()
            current_lr = learning_rate_schedule(
                step,
                config["learning_rate"],
                config["learning_rate"] * 0.1,
                int(total_steps * 0.05),
                total_steps
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            
            
                input, targets = data_get_batch(
                        dataset=np_set,
                        batch_size=config["batch_size"],
                        context_length=config["context_length"],
                        device=device,
                        dtype=torch.int32
                    )
            
            
                prop = transformer(input)
                loss = customm.cross_entropy(prop, targets)
                loss.backward() # compute gradients 
                
                gradient_clipping(transformer.parameters(), 1.0) # clip gradients 
                
                optimizer.step()
            
            if step % 3000 == 0:
                customm.save_checkpoint(transformer, optimizer, step, f'data/tinystories{id}_32000_{step}.pt')
            if step % 20 == 0:
                run.log({"current loss" : loss, "lr" : current_lr, "step" : step})
            print("Finished step:", step)
            pass
    run.finish()
    return

def main(**kwargs):
    
    model_index = kwargs['idx'] if 'idx' in kwargs else 1
    learning_rate = kwargs['lr'] if 'lr' in kwargs else 0.01
    batch_size = kwargs['bs'] if 'bs' in kwargs else 512
    beta1 = kwargs['beta1'] if 'beta1' in kwargs else 0
    beta2 = kwargs['beta2'] if 'beta2' in kwargs else 0.999
    weight_decay = kwargs['wd'] if 'wd' in kwargs else 0.01
    use_rope = not kwargs['nope'] if 'nope' in kwargs else True
    use_swiglu = kwargs['swiglu'] if 'swiglu' in kwargs else False
    config = {
        "id" : model_index,
        "src" : "data/d3_enc_max_32000.npy",
        "num_training_epochs" : 3,
        "batch_size" : batch_size,
        "context_length" : 256,
        "vocab_size" : 32000,
        "num_layers" : 4,
        "num_heads" : 16,
        "d_model" : 512,
        "d_ff" : 1344,
        "theta" : 10000.0,
        "beta1" : beta1,
        "beta2" : beta2,
        "weight_decay" : weight_decay,
        "learning_rate" : learning_rate,
        "lrmin" : 0.00,
        "lrmax" : 0.00,
        "t_warmup_t" : 0.00,
        "t_cosine_c" : 0.00,
        "dtype" : torch.bfloat16,
        "checkpoint" : torch.int32,
        "use_rope" : use_rope,
        "use_swiglu" : use_swiglu,
    }
    run = wandb.init(
        project="tiny_stories_model",
        config= config
    )
    return training_loop(config, run)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx', type=int, default=0, help="model index")
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('-bs', type=int, default=512, help="batch size")
    parser.add_argument('-beta1', type=float, default=0.9, help="beta1")
    parser.add_argument('-beta2', type=float, default=0.999, help="beta2")
    parser.add_argument('-wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('-nope', type=bool, default=False, help="Do not use RoPE")
    parser.add_argument('-swiglu', type=bool, default=False, help="Use SwiGLU instead of ReLU")
    main(
        idx=parser.parse_args().idx,
        lr=parser.parse_args().lr,
        bs=parser.parse_args().bs,
        beta1=parser.parse_args().beta1,
        beta2=parser.parse_args().beta2,
        wd=parser.parse_args().wd,
        nope=parser.parse_args().nope,
        swiglu=parser.parse_args().swiglu,
    )