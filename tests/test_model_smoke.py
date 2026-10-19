"""Forward, guardado y carga del modelo con un encoder diminuto (requiere red la primera vez)."""
import pytest
import torch

from schema import Schema

TINY = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def tiny():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # sin red o sin caché
        pytest.skip(f"no se pudo cargar {TINY}: {e}")
    return TINY


@pytest.mark.parametrize("use_crf", [False, True])
def test_forward_save_load(tiny, tmp_path, use_crf):
    from model_in import JointBERT
    s = Schema.from_catalog()
    m = JointBERT(s, model_name_or_path=tiny, use_crf=use_crf).eval()  # sin dropout para comparar
    B, L = 2, 8
    ids = torch.randint(5, 100, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    heads = torch.zeros(B, len(s.heads), dtype=torch.long)
    slots = torch.zeros(B, L, dtype=torch.long)
    slots[:, 0] = -100
    out = m(ids, mask, torch.zeros_like(ids), heads, slots)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert set(out.head_logits) == set(s.heads)
    assert all(out.head_logits[h].shape == (B, 2) for h in s.heads)
    assert out.slot_logits.shape == (B, L, len(s.slot_labels))
    assert len(m.decode_slots(out.slot_logits, mask)[0]) == L

    m.save(tmp_path)
    m2 = JointBERT.load(tmp_path)
    assert m2.schema.to_dict() == s.to_dict()
    out2 = m2(ids, mask, torch.zeros_like(ids))
    assert torch.allclose(out.head_logits["activ_cls"], out2.head_logits["activ_cls"], atol=1e-5)
