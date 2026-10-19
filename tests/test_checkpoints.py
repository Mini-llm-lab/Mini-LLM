from tokenizers import Tokenizer, models as tokenizer_models

from mini_llm.checkpoints import load_tokenizer


def make_tok(path, extra=None):
    vocab = {"<eos>": 0, **(extra or {})}
    tok = Tokenizer(tokenizer_models.WordLevel(vocab, unk_token="<eos>"))
    tok.save(str(path))
    return tok


def test_load_tokenizer_walks_up_to_the_lineage_folder(tmp_path):
    lineage = tmp_path / "32k"
    ckpt = lineage / "40M" / "mini-llm-base-40M-1steps.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    make_tok(lineage / "tok.json")
    assert load_tokenizer(ckpt).get_vocab_size() == 1


def test_load_tokenizer_prefers_the_sibling_tok(tmp_path):
    lineage = tmp_path / "32k"
    ckpt = lineage / "40M" / "mini-llm-base-40M-1steps.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    make_tok(lineage / "tok.json")
    make_tok(ckpt.parent / "tok.json", extra={"a": 1})
    assert load_tokenizer(ckpt).get_vocab_size() == 2
