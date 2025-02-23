import math

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715
MAX_STEPS = 19073

def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * (it+1) / WARMUP_STEPS
    if it > MAX_STEPS:
        return MIN_LR
    
    deacy_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= deacy_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi *deacy_ratio))
    return MAX_LR + coeff * (MAX_LR - MIN_LR) 

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr_ka(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    print(get_lr(1) == get_lr_ka(1))
    print(get_lr(25)== get_lr_ka(25))
    print(get_lr(51)== get_lr_ka(51))