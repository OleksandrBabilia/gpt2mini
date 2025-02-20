import math

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 50

def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * (it+1) / WARMUP_STEPS
    if it > MAX_STEPS:
        return MIN_LR
    
    deacy_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= deacy_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi *deacy_ratio))
    return MAX_LR + coeff * (MAX_LR - MIN_LR) 

if __name__ == "__main__":
    print(get_lr(1))
    print(get_lr(25))
    print(get_lr(51))