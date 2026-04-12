# Flash-Attention Verification Tests

After installing the environment (`install_verl_hpc.sh`) and activating it
(`activate_verl_hpc.sh`), run these tests in order to verify flash-attn works.

## 1. Basic import and version

```bash
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
```

## 2. CUDA extension compiled correctly (most common failure point)

```bash
python -c "import flash_attn_2_cuda; print('CUDA ext OK:', flash_attn_2_cuda)"
```

If this gives `ImportError` or `undefined symbol`, flash-attn and the current
torch/CUDA ABI are mismatched.

## 3. Forward pass (minimal runnable example)

```bash
python -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device='cuda')
out = flash_attn_func(q, k, v, causal=True)
print('forward OK:', out.shape, out.dtype, out.device)
"
```

## 4. Backward pass (confirms training works, not just inference)

```bash
python -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 512, 8, 64, dtype=torch.float16, device='cuda', requires_grad=True)
k = torch.randn(2, 512, 8, 64, dtype=torch.float16, device='cuda', requires_grad=True)
v = torch.randn(2, 512, 8, 64, dtype=torch.float16, device='cuda', requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
print('backward OK, grad norm:', q.grad.norm().item())
"
```

## 5. Numerical comparison with PyTorch SDPA

```bash
python -c "
import torch, torch.nn.functional as F
from flash_attn import flash_attn_func
torch.manual_seed(0)
B, S, H, D = 2, 256, 8, 64
q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
k = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
v = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
o_fa = flash_attn_func(q, k, v, causal=True)
# SDPA expects (B, H, S, D)
o_ref = F.scaled_dot_product_attention(
    q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
).transpose(1,2)
diff = (o_fa - o_ref).abs().max().item()
print('max abs diff vs SDPA:', diff)
assert diff < 1e-2, 'mismatch!'
print('numerical check OK')
"
```

## 6. Confirm transformers uses flash-attn

```bash
python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained(
    'sshleifer/tiny-gpt2',
    torch_dtype=torch.float16,
    attn_implementation='flash_attention_2',
).cuda()
print('attn_implementation:', m.config._attn_implementation)
print('transformers + flash-attn OK')
"
```

## Summary

| Test | Pass means |
|------|-----------|
| 1-2  | Installed correctly |
| 3-4  | Forward and backward work on GPU |
| 5    | Results are numerically correct |
| 6    | transformers integration works |
