import numpy as np  
import torch
import torch.nn as nn
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import BPETrainer
import cs336_basics.transformer.modules as customm
from cs336_basics.transformer.modules import TransformerLM, AdamW, learning_rate_schedule, data_get_batch
import pickle as pkl
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from tqdm.auto import tqdm


def main():
    
    vocab_size = 32000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if vocab_size == 32000:
        tokenizer = BPETokenizer.from_files("data/bpe_vocab_max.pkl", "data/bpe_merges_max.pkl")
    else:
        tokenizer = BPETokenizer.from_files("data/bpe_vocab_max.pkl", "data/bpe_merges_max.pkl")
    
    config = {
        "src" : "data/d4_enc_max_32000.npy",
        "num_training_epochs" : 2,
        "batch_size" : 128,
        "context_length" : 256,
        "vocab_size" : 32000,
        "num_layers" : 4,
        "num_heads" : 16,
        "d_model" : 512,
        "d_ff" : 1344,
        "theta" : 10000.0,
        "beta1" : 0.9,
        "beta2" : 0.999,
        "weight_decay" : 0.01,
        "learning_rate" : 3e-4,
        "lrmin" : 0.00,
        "lrmax" : 0.00,
        "t_warmup_t" : 0.00,
        "t_cosine_c" : 0.00,
        "dtype" : torch.bfloat16,
        "checkpoint" : torch.int32,
    }
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
        use_rope = True,
        use_swiglu = True,
    )
    transformer = transformer.to(device)
    checkpoint = torch.load("data/tinystories72_32000_12000.pt", weights_only=True)
    transformer.load_state_dict(checkpoint["m_state_dict"]) # loading checkpoint 
    print("checpoint loaded")
    np_set = np.load(config["src"], mmap_mode='r')
    len_dataset = np_set.shape[0]
    batch_size = config["batch_size"]
    loss_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len_dataset, batch_size*20)):
            input, targets = data_get_batch(
                dataset=np_set,
                batch_size=batch_size,
                context_length=config["context_length"],
                device=device,
                dtype=torch.int32
            )
            # Make sure input and targets have correct dtype for embedding
            input = input.to(device=device, dtype=torch.long)
            targets = targets.to(device=device, dtype=torch.long)

            logits = transformer(input)
            loss = customm.cross_entropy(logits, targets)
            
            loss_list.append(loss.item())  # detach to Python scalar

    loss_complete = sum(loss_list) / len(loss_list)
    print(f'Full loss: {loss_complete}')
    return

if __name__ == "__main__":
    main()