import pytest
import torch

from mini_llm.config import Cfg
from mini_llm.pretrain import kd_loss, validate_teacher
from mini_llm.teacher import to_native_state_dict


def test_kd_loss_is_zero_for_equal_logits():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    assert kd_loss(logits, logits.clone(), 2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_loss_chunking_matches_full():
    torch.manual_seed(0)
    s = torch.randn(4, 8, 11)
    t = torch.randn(4, 8, 11)
    full = kd_loss(s, t, 1.7, chunk=10_000).item()
    for chunk in (1, 3, 16):
        assert kd_loss(s, t, 1.7, chunk=chunk).item() == pytest.approx(full, rel=1e-5)


def test_kd_loss_backward_reaches_student():
    s = torch.randn(2, 3, 5, requires_grad=True)
    t = torch.randn(2, 3, 5)
    kd_loss(s, t, 2.0).backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_validate_teacher():
    validate_teacher(Cfg(vocab=32768, ctx=512), Cfg(vocab=32768, ctx=512))
    with pytest.raises(SystemExit):
        validate_teacher(Cfg(vocab=16384, ctx=512), Cfg(vocab=32768, ctx=512))
    with pytest.raises(SystemExit):
        validate_teacher(Cfg(vocab=32768, ctx=256), Cfg(vocab=32768, ctx=512))


def test_to_native_state_dict():
    emb = torch.zeros(4, 2)
    sd = to_native_state_dict(
        {"emb.weight": emb, "norm.w": torch.ones(2), "cos": torch.ones(2, 1), "sin": torch.ones(2, 1)}
    )
    assert set(sd) == {"emb.weight", "norm.w", "head.weight"}
    assert sd["head.weight"] is emb
