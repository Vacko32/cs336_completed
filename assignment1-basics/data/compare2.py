from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import BPETrainer
import cs336_basics.transformer.modules as customm
import pickle as pkl
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")



owt_fp = "data/owt_10_samples.txt"
ts_fp = "data/tinystories_10_samples.txt"


merges = "data/bpe_merges_max_new.pkl"
merges_2 = "data/bpe_merges.pkl"
vocab_2 = "data/bpe_vocab.pkl"
vocab = "data/bpe_vocab_max_new.pkl"

print(type(merges))
print(type(vocab))




sizeowt = os.path.getsize(owt_fp)
sizety = os.path.getsize(ts_fp)

print("=========================")
print("   Dataset Information   ")
print("=========================")
print("   Ten samples of OWT    ")
print("=========================")
print("OWT file size:", sizeowt, "bytes")

print("=========================")
print("   Dataset Information   ")
print("=========================")
print("   Ten samples of TS    ")
print("=========================")
print("TinyStories file size:", sizety, "bytes")


tokenizer_owt = BPETokenizer.from_files(vocab, merges, special_tokens=["<|endoftext|>"])
tokenizer_ts = BPETokenizer.from_files(vocab_2, merges_2, special_tokens=["<|endoftext|>"])


with open(owt_fp, "r") as f:
    owt_str = f.read()
    print("encoding sample of owt")

    encoded_otw = tokenizer_owt.encode(owt_str)
    encoded_ts = tokenizer_ts.encode(owt_str)
    encoded_gpt = tokenizer(owt_str)["input_ids"]
    print("encoding finshed")
    otw_res =  sizeowt /len(encoded_otw) 
    ts_res =   sizeowt / len(encoded_ts) 
    gpt2_res = sizeowt /len(encoded_gpt)
    print("OTW results", otw_res)
    print("TS results", ts_res)
    print("GPT2 results", gpt2_res)

with open(ts_fp, "r") as f:
    owt_str = f.read()
    print("encoding sample of TS")

    encoded_otw = tokenizer_owt.encode(owt_str)
    encoded_ts = tokenizer_ts.encode(owt_str)
    encoded_gpt = tokenizer(owt_str)["input_ids"]

    print("comparation")
    print("gpt", encoded_gpt[0:10])
    print("gpt", encoded_otw[0:10])
    print("encoding finshed")
    otw_res = sizety / len(encoded_otw) 
    ts_res = sizety / len(encoded_ts) 
    gpt2_res = sizety / len(encoded_gpt)
    print("OTW results", otw_res)
    print("TS results", ts_res)
    print("GPT2 results", gpt2_res)



print("=========== Preparing OTW and tinystories dataset with gpt vocab and merges =========")


"""
d1 = "data/otw_train.txt"
d2 = "data/otw_test.txt"
d4 = "data/TinyStoriesV2-GPT4-valid.txt"
d3 = "data/TinyStoriesV2-GPT4-train.txt"

d1_enc = tokenizer(d1)["input_ids"]
np.save("data/d1_enc.npy", np.array(d1_enc, dtype=np.uint16))
d2_enc = tokenizer(d2)["input_ids"]
np.save("data/d2_enc.npy", np.array(d2_enc, dtype=np.uint16))
d3_enc = tokenizer(d3)["input_ids"]
np.save("data/d3_enc.npy", np.array(d3_enc, dtype=np.uint16))
d4_enc = tokenizer(d4)["input_ids"]
np.save("data/d4_enc.npy", np.array(d4_enc, dtype=np.uint16))
"""
print("END OF SCRIPT")
