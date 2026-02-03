#!/usr/bin/env python3
"""Generate reference values for FMHA forward test."""

import numpy as np
from scipy.special import softmax

# Test parameters
BATCH = 2   # batch size
NHEAD = 1   # number of heads
M = 24      # seqlen_q
N = 32      # seqlen_k
K = 8       # hdim_q
O = 16      # hdim_v (different from K)
scale_s = 1.0
SEED = 42   # Fixed seed for reproducibility

# Generate random inputs with fixed seed
# Layout: [batch, nhead, seqlen, hdim]
np.random.seed(SEED)
q = np.random.uniform(-0.5, 0.5, (BATCH, NHEAD, M, K)).astype(np.float32)
k = np.random.uniform(-0.5, 0.5, (BATCH, NHEAD, N, K)).astype(np.float32)
v = np.random.uniform(-0.5, 0.5, (BATCH, NHEAD, N, O)).astype(np.float32)

print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)

# Compute attention per batch/head: O = softmax(Q @ K^T * scale_s) @ V
output = np.zeros((BATCH, NHEAD, M, O), dtype=np.float32)
for b in range(BATCH):
    for h in range(NHEAD):
        scores = np.matmul(q[b, h], k[b, h].T) * scale_s  # [M, N]
        attn = softmax(scores, axis=-1)                    # [M, N]
        output[b, h] = np.matmul(attn, v[b, h])            # [M, O]

def print_cpp_array(name, arr):
    flat = arr.flatten()
    print(f"\n// C++ array ({len(flat)} values):")
    print(f"const float {name}[] = {{")
    for i in range(0, len(flat), 8):
        line = ", ".join(f"{flat[i+j]:.6f}f" for j in range(min(8, len(flat) - i)))
        print(f"    {line},")
    print("};")

# Print input arrays for C++
print_cpp_array("q_data", q)
print_cpp_array("k_data", k)
print_cpp_array("v_data", v)
print_cpp_array("numpy_expected", output)

# Print first/last for quick comparison
print("\nFirst 5 outputs:", output.flatten()[:5])
print("Last 5 outputs:", output.flatten()[-5:])

# Print test parameters for reference
print(f"\n// Problem dimensions:")
print(f"// BATCH={BATCH}, NHEAD={NHEAD}, M={M}, N={N}, K={K}, O={O}")
