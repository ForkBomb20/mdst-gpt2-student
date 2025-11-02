import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) Return min if past decay horizon
    if it > max_steps:
        return min_lr
    # 3) Cosine decay
    decay_ration = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ration <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ration))
    return min_lr + coeff * (max_lr - min_lr)