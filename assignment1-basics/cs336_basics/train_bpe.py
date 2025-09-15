import regex as re
import cProfile, pstats
import time
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries
import psutil
import heapq
from .logger import Logger
import sys
from collections import Counter, defaultdict




class BPETrainer:
    def __init__(self, file_path : str, vocab_size : int, special_tokens : list = []):
        self._file_p = file_path
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens
        self._special_tokens = sorted(self._special_tokens, key=len)
        self._was_trained = False
        self.merges : list[tuple[bytes, bytes]] = []
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def to_byte(self, x):
                if isinstance(x, int):
                    return bytes([x])   # int → single byte
                return x                # already bytes → keep as-is
    def process_chunk(self, args):
        chunk_id, chunk_start, chunk_end = args

        tok_re = re.compile("|".join(map(re.escape, self._special_tokens)))
        pat_re = self.pat  # already compiled in __init__

        word_to_splits = {}
        word_freq = Counter()
        pair_to_words = defaultdict(set)
        pair_freq = Counter()
        with open(self._file_p, "rb") as f:
            f.seek(chunk_start)
            main_chunk = f.read(chunk_end - chunk_start).decode("utf-8")

        start = 0
        for inner in tok_re.finditer(main_chunk):
            if inner.start() > start:
                chunk = main_chunk[start:inner.start()]
                for inner2 in pat_re.finditer(chunk):
                    g = inner2.group()
                    if not g:
                        continue
                    t_bytes = g.encode("utf-8")
                    word_freq[t_bytes] += 1

                    prev = None
                    splits = []
                    for by in t_bytes:            
                        b1 = bytes([by])
                        splits.append(b1)
                        if prev is not None:
                            p = (prev, b1)
                            pair_freq[p] += 1
                            pair_to_words[p].add(t_bytes)
                        prev = b1
                    word_to_splits[t_bytes] = splits
            start = inner.end()

        # Tail 
        if start < len(main_chunk):
            chunk = main_chunk[start:]
            for inner2 in pat_re.finditer(chunk):
                g = inner2.group()
                if not g:
                    continue
                t_bytes = g.encode("utf-8")
                word_freq[t_bytes] += 1

                prev = None
                splits = []
                for by in t_bytes:
                    b1 = bytes([by])
                    splits.append(b1)
                    if prev is not None:
                        p = (prev, b1)
                        pair_freq[p] += 1
                        pair_to_words[p].add(t_bytes)
                    prev = b1
                word_to_splits[t_bytes] = splits

        return word_to_splits, word_freq, pair_to_words, pair_freq



    def run_train_interface(self):
        vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self._vocab_size - 256 - len(self._special_tokens)

        with open(self._file_p, "rb") as f:
            num_processes = 16
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        word_to_splits = {}
        pair_freq = Counter()
        pair_to_words = {}
        word_freqs = Counter() 
        

        inputs = [(i, start, end) for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        with Pool(num_processes) as p:
            outputs = p.map(self.process_chunk, inputs) # word_to_splits, word_freq, pair_to_words, pair_freq is one output
        for w_splits, w_freq, p_words, p_freq in outputs:

            word_freqs.update(w_freq)
            word_to_splits.update(w_splits)
            pair_freq.update(p_freq)
            for pair, words in p_words.items():
                pair_to_words.setdefault(pair, set()).update(words)

        heap = []
        for pair, count in pair_freq.items():
            heapq.heappush(heap, (-count, pair))
        heapq.heapify(heap)
        # Merge loop    
        for n in range(num_merges):
            cand = []
            while heap:
                neg_count, pair = heapq.heappop(heap)
                count = -neg_count
                actual_count = pair_freq.get(pair, 0)
                if actual_count == count:
                    cand = [(neg_count, pair)]
                    lower = None
                    while heap:
                        n_c, b_p = heapq.heappop(heap)
                        if -n_c < count:
                            lower = (n_c, b_p)  # first lower-count item; push back later
                            break
                        if pair_freq.get(b_p, 0) == -n_c:
                            cand.append((n_c, b_p))
                    QQ, best_pair = max(cand, key=lambda x: x[1])
                    for n_c, b_p in cand:
                        if b_p != best_pair:
                            heapq.heappush(heap, (n_c, b_p))
                    if lower is not None:
                        heapq.heappush(heap, lower)

                    pair = best_pair 
                    break
            else:
                break
            a, b = pair[0], pair[1]
            new_token_bytes = bytes(pair[0]) + bytes(pair[1])
            vocab[256 + n] = new_token_bytes
            
            self.merges.append((a, b))
            #print("Merged:", pair, "count:", QQ)

            # we look up all words we have 
            affected_words = list(pair_to_words.get(pair, set()))
            for word in affected_words:
                word_freq = word_freqs[word]
                word_splits = word_to_splits[word]
                i = 0 
                while i < len(word_splits) - 1:
                    if word_splits[i] == pair[0] and word_splits[i + 1] == pair[1]:
                        word_splits[i] = new_token_bytes
                        word_splits.pop(i + 1)
                        if pair in pair_freq:
                            del pair_freq[pair]
                        if i > 0:
                            new_pair_left = (word_splits[i-1], new_token_bytes)
                            old_pair_left = (word_splits[i-1], a)
                            # we update ....
                            pair_to_words.setdefault(new_pair_left, set()).add(word)
                            
                            pair_freq[new_pair_left] = pair_freq.get(new_pair_left, 0) + word_freq

                            heapq.heappush(heap, (-pair_freq[new_pair_left], new_pair_left))
                            if old_pair_left in pair_freq:
                                pair_freq[old_pair_left] -= word_freq
                                if pair_freq[old_pair_left] <= 0:
                                    del pair_freq[old_pair_left]
                                else:
                                    heapq.heappush(heap, (-pair_freq[old_pair_left], old_pair_left))
                        if i < len(word_splits) - 1:
                            new_pair_right = (new_token_bytes, word_splits[i + 1])
                            old_pair_right = (b, word_splits[i + 1])

                            # update pair_to_words
                            pair_to_words.setdefault(new_pair_right, set()).add(word)

                            # update pair_freq for the new pair
                            pair_freq[new_pair_right] = pair_freq.get(new_pair_right, 0) + word_freq
                            heapq.heappush(heap, (-pair_freq[new_pair_right], new_pair_right))
                            # decrement old pair frequency and clean up if necessary
                            if old_pair_right in pair_freq:
                                pair_freq[old_pair_right] -= word_freq
                                if pair_freq[old_pair_right] <= 0:
                                    del pair_freq[old_pair_right]
                                else:
                                    heapq.heappush(heap, (-pair_freq[old_pair_right], old_pair_right))
                            
                    else:
                        i += 1

                    
        for s in self._special_tokens:
            vocab[len(vocab)] = s.encode("utf-8")
        return vocab, self.merges

    
    def run_train_interface_cprofile(self, sort_by: str = "cume", lines: int = 20):
        pr = cProfile.Profile()
        pr.enable()
        out = self.run_train_interface()
        pr.disable()
        pstats.Stats(pr).sort_stats(sort_by).print_stats(lines)
        return out
