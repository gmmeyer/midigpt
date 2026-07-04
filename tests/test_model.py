"""Model sanity: shapes, causality, loss masking, tiny-overfit smoke."""

import torch

from midigpt.model import GPTConfig, MusicGPT

TINY = GPTConfig(vocab_size=97, block_size=32, n_layer=2, n_head=2,
                 n_kv_head=1, n_embd=64)


def test_forward_shapes_and_loss():
    m = MusicGPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (3, 32))
    logits, loss = m(x, targets=x)
    assert logits.shape == (3, 32, TINY.vocab_size)
    assert loss.item() > 0


def test_causality():
    m = MusicGPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 32))
    with torch.no_grad():
        a, _ = m(x)
        y = x.clone()
        y[0, 20:] = (y[0, 20:] + 1) % TINY.vocab_size
        b, _ = m(y)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-5)
    assert not torch.allclose(a[0, 20:], b[0, 20:], atol=1e-5)


def test_loss_mask():
    m = MusicGPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (2, 32))
    with torch.no_grad():
        _, full = m(x, targets=x)
        _, ones = m(x, targets=x, loss_mask=torch.ones(2, 32))
        half_mask = torch.zeros(2, 32)
        half_mask[:, :16] = 1.0
        _, half = m(x, targets=x, loss_mask=half_mask)
    assert torch.allclose(full, ones, atol=1e-6)
    assert not torch.allclose(full, half, atol=1e-4)


def test_tiny_overfit():
    torch.manual_seed(0)
    m = MusicGPT(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    pattern = torch.arange(32) % 7  # deterministic repeating sequence
    x = pattern.unsqueeze(0).repeat(8, 1)
    for _ in range(150):
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
