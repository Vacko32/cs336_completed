# Implementation of BPETrainer, BPETokenizer, Transformer "from scratch" 
This repository contains the assigment for cs336 class, which is being lectured at Standford university as one of the grad courses.
After seeing the assignment, I will provide my implementation details, with results, optimalization and also answers to their openended questions



## My solution

### Byte-pair encoding, training tokenizer.

Tokenization is important process when it comes to LLM. We try to represent input features, such as strings, pixel values etc. as a list of integers, ranging from 0 to size of our vocab. 

Vocab size has to be carefully chosen, thanks to it we can make certain tokens represent multiple bytes essentially saving space.

Because we want to save as much memory as possible, the vocabulary needs to be trained on some dataset, where we look for bytes/tokens that are the most frequent neighbours and merged them into a new token. This process is done till we fullfil our predefined vocab_size.

My implementation in cs336_basics/trainer_bpe.py features a class BPETrainer which allows you to train your own vocab and merges on given text. The class enables parallel processing of input text, resulting in very good performance even on larger files (10+gb, see later).


The trainer has been optimalized by these techniques: 
- Uses heap to find the highest frequency pair, which reduces the time complexity from O(n), if we seach through whole hashtable
- Keeps track of every "word" only once in a set datastructure, and storing it's count. This drastically saves memory and speed, because a word that has appeared in text more than 1000 times, is still processed only once
- When merging a pair of bytes/tokens, we can get all the affected_words by a look-up into our hashtable that maps pairs to a set of words.

![alt text](schema1.PNG)


During the training of my own vcab and merges i managed to get these results.

- Training on TinyStories dataset (2gb), for vocab 10000, my interface took **6.38 minutes**, on 8cpus, no gpus, while using parallel preprocessing.
- Training on OpenWebText dataset (10gb+) for vocab 50000, my interface took around **26 minutes**, with 16 cpus and RAM usage under 100gb. 

Shown and the table below, u can see a comparasion of what i trained in contrast to gpt2 tokenizer with vocab size 50,267. 

The table shows the average value of how many bytes does a single token represents from a randomly chosen samples.


| Dataset      | File Size (bytes) | OTW Pretrained | TS Pretrained | GPT-2 Pretrained |
|--------------|-------------------|----------------|---------------|------------------|
| OWT          | 31,487            | **4.8189**     | 3.1892        | 4.7143           |
| TinyStories  | 7,435             | 4.0430         | **4.1146**    | 4.0276           |






My implementation in cs336_basics/tokenizer.py features a class BPETokenizer that is able to encode and decode certain string with encode method, or encode larger files with encode_iterable method.
(Note: Still working on the encode_iterable function which is not optimalized to it's best and still takes a lot oftime. On roughly 11gb files without any parallization we are looking at 3-4 hour encode time)



### Transformer from scratch 

In `cs336_basics/transformers/modules` u can find these functions implemented from scratch that secures succesfull forward pass:

- Linear class
- Embeeding class
- RMSnorm
- SwiGLU class
- RoPE class
- softmax function
- scaled_attention function (Supporting custom masks)
- MultiheadSelfAttentionClass, with and without RoPE
- Transformer Block class, with Two RMSnorm as prenorms, One MHA and one SwiGLU network
- TransformerLM class, featuring the same architecture of transformer block as defined above, supporting checkpointing.

All of the functions and class above supports batch_processing.


The scheme of the implemented transformer from scratch is visible on the scheme below:



![alt text](scheme2.PNG)


This architecture is a little bit different than the one introduced in "Attention is all u need" (https://arxiv.org/abs/1706.0376). First it uses pre-norms instead of post-norms, which is very common in modern architectures.

Examples of generated text on tinystories dataset: 
``Tom went to the box and opened it. Inside, he found a big, red ball. He was very happy. Tom wanted to play with the ball, but he did not know how. He tried to push the ball, but it was too heavy. Tom was sad.
Then, a little girl named Sue came to help Tom. She said, "I can help you, Tom!" Tom was happy. He played with the ball and had lots of fun. Tom was not grumpy anymore. He was happy to have a new friend.
<|endoftext|>``


``But then, Tom said the toy truck, "Please dove into a big box of matches. Let's go find a match to play in."
They looked all day, until Tom saw another big match with a match. He asked all their friend if the box had been too small to fit inside the box! The small girl said no. The small girl said sorry for the small toy, so it was okay and said yes.
They played all morning, and then they both went outside to see their toy box with their toys. When she was tired, they sat down under some big tree, feeling very proud of what was on their adventure, and their fun game had. And she knew it could bring joy to her friends.``


The training was done only for an hour on L40 and A100 on tinystories 2gb dataset. 

Further experiments might be done later on h100/h200 cards, after semester is over.

## CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

### Setup

#### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
