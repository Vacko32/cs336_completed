import numpy as np  
import torch
import torch.nn as nn
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import BPETrainer
from cs336_basics.transformer.modules import TransformerLM, AdamW, learning_rate_schedule, data_get_batch
import pickle as pkl
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from tqdm.auto import tqdm
def top_k_sampling(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(topk_logits, dim=-1)

    
    sampled = torch.multinomial(probs, num_samples=1)
    next_token = torch.gather(topk_indices, -1, sampled).squeeze(-1)

    if squeeze_back:
        return next_token.squeeze(0)
    return next_token

def nucleus_sampling(logits, p: float = 0.9):

    # Greedy: pick the index of the max logit
    _, original_indices = torch.max(logits, dim=-1, keepdim=True)
    return original_indices.squeeze(-1)


def main():
    print("got into main")
    vocab_size = 32000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if vocab_size == 32000:
        tokenizer = BPETokenizer.from_files("data/bpe_vocab_max.pkl", "data/bpe_merges_max.pkl")
    else:
        tokenizer = BPETokenizer.from_files("data/bpe_vocab_max.pkl", "data/bpe_merges_max.pkl")
    
    config = {
        "src" : "data/d3_enc_max_32000.npy",
        "num_training_epochs" : 2,
        "batch_size" : 512,
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
    context_tokens = []
    
    prompt = open("data/prompt_file", "r").read()
    print("prompt read")    
    
    print("encoded prompt:", prompt)
    end_token_string = "<|endoftext|>"
    end_token = tokenizer.encode(end_token_string)

    
    """
    for token in encodings[-1:-255]:
        context_tokens.append(token)
    """
    temperature = 1.2
    max_len = 256
    final_story = tokenizer.encode(prompt)
    with torch.no_grad():
        for i in range(max_len):
            prompt = tokenizer.encode(prompt)
            encodings = prompt
            context_tokens = encodings[-255:]
            if len(context_tokens) < 1:
                context_tokens = [0]
            print("Current context tokens:", context_tokens)
            input_ids = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = transformer(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = top_k_sampling(probs).item()
            final_story.append(next_token)
            context_tokens.append(next_token)
            prompt = tokenizer.decode(context_tokens)
            print("Current prompt:", prompt)
            if next_token == end_token[0]:
                break
        print("Generated text:")
        print(prompt)



    
    print("sucessfull forward pass")
    print(tokenizer.decode(final_story))
    return



if __name__ == "__main__":
    main()