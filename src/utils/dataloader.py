import torch
import tiktoken
import os
import numpy as np


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=True, data_path_file=None, data_path_folder=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shards = None
        assert split in {"train", "val"}

        if data_path_folder:
            shards = os.listdir(data_path_folder)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_path_folder, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"No shards found for split {split}"
            if master_process:
                print(f"Found {len(shards)} shards for split {split}")
            
            self.current_shard = 0
            self.tokens = self.load_tokens[self.shards[self.current_shard]]
            

        else:
            with open(data_path_file, "r") as file:
                text = file.read()
            
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens)

            print(f"Loaded {len(self.tokens)} tokens")
            print(f"1 epoch {len(self.tokens) // (B*T)} batches")

        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        
        x = (buf[:-1].view(self.B, self.T))
        y = (buf[1:].view(self.B, self.T))

        self.current_position += self.B * self.T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.shards:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        
        return x, y
    
    def load_tokens(filename):
        npt = np.load(filename)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt