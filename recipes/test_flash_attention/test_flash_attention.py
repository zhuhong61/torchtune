import torch
import numpy as np
import random
import math
from torch import nn
import intel_extension_for_pytorch

from torchtune.modules import RotaryPositionalEmbeddings

packed = True

bs = 4
num_head = 32
max_seq_len = 512
head_dim = 128
mask = torch.zeros((bs, max_seq_len, max_seq_len), dtype=bool)
input_pos = np.zeros((bs, max_seq_len))
if packed:
    num_seq_packed = 4
    for idx in range(bs):
        random.seed(idx)
        start_seq_packed = np.sort(np.append(random.sample(range(0,max_seq_len), num_seq_packed), 0))
        mask_idx = []
        input_pos_idx = []
        for i, start in enumerate(start_seq_packed):
            seq_len = start_seq_packed[i+1]-start if i<num_seq_packed else max_seq_len-start
            mask_packing = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            mask_idx.append(mask_packing)
            input_pos_idx.extend(list(range(seq_len)))
        mask_idx = torch.block_diag(*mask_idx)
        mask[idx, :, :] = mask_idx
        input_pos[idx, :] = input_pos_idx
    mask = mask[:, None, :, :].xpu()
else:
    mask = None
    input_pos = None

rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
output_proj = nn.Linear(head_dim*num_head, head_dim*num_head, bias=False).xpu()

# q [batchsize, num_head, seqlen, head_dim] 
torch.manual_seed(20240711)
q = torch.randint(0, 2^32, [bs, max_seq_len, num_head, head_dim])
k = torch.randint(0, 2^32, [bs, max_seq_len, num_head, head_dim])
v = torch.randint(0, 2^32, [bs, max_seq_len, num_head, head_dim])

# qkv = np.load('batchdata/qkv.npz')
# q = torch.from_numpy(qkv['q'])
# k = torch.from_numpy(qkv['k'])
# v = torch.from_numpy(qkv['v'])
# mask = torch.from_numpy(qkv['mask'])
# mask = mask[:, None, :, :].xpu()
# input_pos = qkv['input_pos']

# position embedding
q = rope(q, input_pos=input_pos)
k = rope(k, input_pos=input_pos)
# [b, n_h, s, h_d]
q = q.transpose(1, 2).bfloat16().xpu()
k = k.transpose(1, 2).bfloat16().xpu()
v = v.transpose(1, 2).bfloat16().xpu()


print(f'q.dtype: {q.dtype}')
attn_bias = torch.zeros_like(mask, dtype=q.dtype)
att_bias = attn_bias.masked_fill_(mask.logical_not(), float('-inf'))

# attention implementation
scale = None
scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
attn_weight += attn_bias
attn_weight = torch.softmax(attn_weight, dim=-1)
output = torch.matmul(attn_weight, v)

# torch scaled_dot_product_attention
# output = torch.nn.functional.scaled_dot_product_attention(
#     q, 
#     k, 
#     v, 
#     attn_mask=att_bias,
#     dropout_p=0,
#     is_causal=att_bias is None
#     )


print(f'output.isnan: {torch.sum(torch.isnan(output))}')
print(output[0][0])
torch.save(output.detach().cpu(), 'batchdata/test_output_xpu.pt')
# output = output.transpose(1, 2).contiguous().view(bs, max_seq_len, -1)
# outout = output_proj(output)
# print(f'output.isnan: {torch.sum(torch.isnan(output))}')
# print(output)
