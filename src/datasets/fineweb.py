import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

LOCAL_DIR = "../../data/edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train")

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    tokens = [EOT]
    tokens.extend(enc.encode_ordinary(doc["text"]))

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dict too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((SHARD_SIZE, ), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < SHARD_SIZE:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index ==0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            shard_index +=1
            progress_bar = None

            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

