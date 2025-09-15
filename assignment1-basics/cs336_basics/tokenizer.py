from __future__ import annotations
from collections.abc import Iterable
from collections.abc import Iterator
import regex as re
import pickle



"""
Class made for tokenizing and decoding strings.

This class already expects pretrained vocab and merges.

Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.


Attributes:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]
    vocablen: int
    _special_tokens_bytes: list[list[bytes]]
    special_tokens_dict: dict[str, int]
    reverse_vocab: dict[bytes, int]


Returns:
    BPETokenizer

"""
class BPETokenizer:
        def __init__(self, vocab, merges, special_tokens=None):
            self.vocab = vocab 
            self.merges = {}
            self.special_tokens = sorted(special_tokens or [], key=len)
            self.encoded_chunks_cache : dict[str : list[int]] = {}
            self.vocablen = len(vocab)
            print("Current VocabLen:", self.vocablen)
            self.reverse_vocab = {v: k for k, v in vocab.items()}  
            self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

            self._special_tokens_bytes: list[list[bytes]] = []
            for token in self.special_tokens:
                token_bytes = token.encode('utf-8')
                self._special_tokens_bytes.append(list(token_bytes))
            self._special_tokens_bytes.sort(key=len, reverse=True)


            current_id = max(vocab.keys()) + 1
            for b1, b2 in merges:
                id1 = self.reverse_vocab[b1]
                id2 = self.reverse_vocab[b2]
                self.merges[(id1, id2)] = current_id
                current_id += 1

            self.special_tokens_dict = {}
            for i, token in enumerate(self.special_tokens):
                token_bytes = token.encode('utf-8')
                token_id = self.vocablen + i - 1
                self.special_tokens_dict[token] = token_id
                self.vocab[token_id] = token_bytes
        @classmethod
        def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
            with open(vocab_filepath, "rb") as v:
                vocab = pickle.load(v)
            with open(merges_filepath, "rb") as m:
                merges = pickle.load(m)
            return cls(vocab, merges, special_tokens)


        def encode(self, text: str) -> list[int]:
            encoded_ids = []
            text_bytes = text.encode('utf-8')
            special_positions: list[tuple[int, int, int]] = []

            for special_bytes in self._special_tokens_bytes:
                search_start = 0
                while True:
                    search_start = text_bytes.find(bytes(special_bytes), search_start)
                    if search_start == -1:
                        break
                    search_end = search_start + len(special_bytes)
                    if special_positions and any(s <= search_end <= e or s <= search_start < e for s, e, _ in special_positions):
                        search_start = search_end
                        continue
                    special_positions.append(
                        (search_start, search_end, self.vocablen + self.special_tokens.index(''.join(map(chr, special_bytes))) - 1)
                    )
                    search_start = search_end
            special_positions.sort(key=lambda x: x[0])
            current_char_index = 0
            for start_byte, end_byte, special_id in special_positions:
                start_char_index = len(text_bytes[:start_byte].decode('utf-8', errors='ignore'))
                chunk_matches = re.finditer(self.pat, text[current_char_index:start_char_index])

                for match in chunk_matches:
                    str_match = match.group()

                    
                    cached = self.encoded_chunks_cache.get(str_match)
                    if cached is not None:
                        encoded_ids.append(self.encoded_chunks_cache.get(str_match))
                    else:
                        chunk_ids = self.encode_chunk(str_match)
                        encoded_ids.extend(chunk_ids)



                encoded_ids.append(special_id)
                current_char_index = len(text_bytes[:end_byte].decode('utf-8', errors='ignore'))

            if current_char_index < len(text):
                chunk_matches = re.finditer(self.pat, text[current_char_index:])
                for match in chunk_matches:
                    str_match = match.group()

                    # look up to our dict of already encoded words 
                    cached = self.encoded_chunks_cache.get(str_match)
                    if cached is not None:
                        encoded_ids.append(self.encoded_chunks_cache.get(str_match))
                    else:
                        chunk_ids = self.encode_chunk(str_match)
                        encoded_ids.extend(chunk_ids)


            #self.encoded_chunks_cache[text] = self.encoded_chunks_cache.get(text) or encoded_ids
            return encoded_ids

        def encode_chunk(self, text: str) -> list[int]:
            if not text:
                return []

            text_bytes = text.encode("utf-8")
            final_ids = []
            byte_index = 0

            while byte_index < len(text_bytes):
                byte_char = bytes([text_bytes[byte_index]])
                token_id = self.reverse_vocab.get(byte_char)
                if token_id is not None:
                    final_ids.append(token_id)
                byte_index += 1

            while True:
                i = 0
                lowest_pair: tuple[int, int] | None = None
                replacement_token: int | None = None
                curr_priority: int | None = None
                while i < len(final_ids) - 1:
                    pair = (final_ids[i], final_ids[i + 1])
                    if pair in self.merges:
                        priority = self.merges[pair] - self.vocablen
                        if curr_priority is None or priority < curr_priority:
                            curr_priority = priority
                            lowest_pair = (i, i + 1)
                            replacement_token = self.merges[pair] - self.vocablen + 256
                    i += 1
                if curr_priority is None:
                    break
                final_ids[lowest_pair[0]] = replacement_token
                final_ids.pop(lowest_pair[1])

            return final_ids

        def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
            for chunk in iterable:
                yield from self.encode(chunk)     
        
        def decode(self, tokens: list[int]) -> str:
            
            result = []
            byte_stream = bytearray()

            for token in tokens:
                if token <= self.vocablen:
                    byte_stream += self.vocab.get(token, b'')
                    
                else:
                    if byte_stream:
                        result.append(byte_stream.decode('utf-8', errors='surrogateescape'))
                        byte_stream = bytearray()
                    result.append(self.special_tokens[token - self.vocablen])

            if byte_stream:
                result.append(byte_stream.decode('utf-8', errors='surrogateescape'))
            
            return ''.join(result)
    