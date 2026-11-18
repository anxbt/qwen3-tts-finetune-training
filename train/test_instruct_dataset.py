"""CPU unit check for instruct_dataset.py. No GPU, no model: stubs the qwen_tts imports and uses a fake tokenizer.

  python train/test_instruct_dataset.py      # uses third_party/qwen_finetuning; set QWEN_FT_DIR to use another copy

Checks: (1) empty instruction == the official collate_fn output, (2) an instruction shifts every row right by L, puts
text-only embeddings in front, keeps labels at -100 there, and moves the speaker slot to 6 + L, (3) mixed batch pads correctly.
"""
import os
import sys
import types

import numpy as np
import torch

# stubs for what dataset.py imports (qwen_tts, librosa); the real mel code is not needed here
for name in ["librosa", "qwen_tts", "qwen_tts.core", "qwen_tts.core.models", "qwen_tts.core.models.configuration_qwen3_tts", "qwen_tts.core.models.modeling_qwen3_tts"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["qwen_tts.core.models.configuration_qwen3_tts"].Qwen3TTSConfig = object
sys.modules["qwen_tts.core.models.modeling_qwen3_tts"].mel_spectrogram = lambda *a, **k: torch.zeros(1, 128, 10)
sys.path.insert(0, os.environ.get("QWEN_FT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "qwen_finetuning")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TTSDataset  # noqa: E402
from instruct_dataset import InstructTTSDataset  # noqa: E402


class FakeProcessor:
    """Role prompt -> [10,11,12] + one id per character + 5 trailing ids; user prompt -> [20,21] + chars + [22,23]."""
    def __call__(self, text, return_tensors=None, padding=None):
        body = text.split("\n", 1)[1]
        if text.startswith("<|im_start|>assistant"):
            body = body.split("<|im_end|>")[0]
            ids = [10, 11, 12] + [100 + ord(c) % 50 for c in body] + [90, 91, 92, 93, 94]
        else:
            body = body.split("<|im_end|>")[0]
            ids = [20, 21] + [200 + ord(c) % 50 for c in body] + [22, 23]
        return {"input_ids": torch.tensor(ids)}


T = types.SimpleNamespace
config = T(tts_pad_token_id=1, tts_bos_token_id=2, tts_eos_token_id=3,
           talker_config=T(codec_nothink_id=4, codec_think_bos_id=5, codec_think_eos_id=6, codec_pad_id=7, codec_bos_id=8, codec_eos_token_id=9))
rows = [{"text": "abcd", "audio_codes": np.arange(5 * 16).reshape(5, 16) % 1000, "ref_audio": "x", "audio": "x"},
        {"text": "efghij", "audio_codes": (np.arange(8 * 16).reshape(8, 16) + 3) % 1000, "ref_audio": "x", "audio": "x"}]


def make(cls, instructs):
    data = [{**r, "instruct": ins} for r, ins in zip(rows, instructs)]
    ds = cls(data, FakeProcessor(), config)
    ds.extract_mels = lambda audio, sr: torch.zeros(1, 4, 128)
    ds._load_audio_to_np = lambda p: (np.zeros(10, dtype=np.float32), 24000)
    return ds


def batch_of(ds):
    return ds.collate_fn([ds[0], ds[1]])


official = batch_of(make(TTSDataset, ["", ""]))
empty = batch_of(make(InstructTTSDataset, ["", ""]))
for k, v in official.items():
    assert torch.equal(v, empty[k]), f"empty instruction differs from the official layout in {k}"
assert empty["spk_pos"].tolist() == [6, 6]
print("1. empty instruction == official collate: ok")

ins = "Speak sadly."
ds = make(InstructTTSDataset, [ins, ""])
item0 = ds[0]
L = item0["instruct_ids"].shape[1]
assert L == 2 + len(ins) + 2, L
b = ds.collate_fn([ds[0], ds[1]])
# row 0 (with instruction): the official row shifted right by L
n = official["attention_mask"][0].sum().item()
assert b["attention_mask"][0].sum().item() == n + L
assert torch.equal(b["input_ids"][0, L:L + n], official["input_ids"][0, :n])
assert torch.equal(b["codec_0_labels"][0, L:L + n], official["codec_0_labels"][0, :n])
assert torch.equal(b["codec_ids"][0, L:L + n], official["codec_ids"][0, :n])
assert torch.equal(b["codec_mask"][0, L:L + n], official["codec_mask"][0, :n])
assert torch.equal(b["codec_embedding_mask"][0, L:L + n], official["codec_embedding_mask"][0, :n])
# the instruction block: text ids, text mask on, codec channel 0, codec mask off, no labels, attention on
assert torch.equal(b["input_ids"][0, :L, 0], item0["instruct_ids"][0])
assert (b["input_ids"][0, :L, 1] == 0).all()
assert b["text_embedding_mask"][0, :L].all() and not b["codec_embedding_mask"][0, :L].any()
assert (b["codec_0_labels"][0, :L] == -100).all() and not b["codec_mask"][0, :L].any() and (b["attention_mask"][0, :L] == 1).all()
assert b["spk_pos"].tolist() == [6 + L, 6]
assert not b["codec_embedding_mask"][0, 6 + L] and b["input_ids"][0, 6 + L, 1] == 0  # speaker slot
print(f"2. instruction (L={L}) shifts the row, text-only embeddings in front, spk_pos {b['spk_pos'].tolist()}: ok")

# 3. row 1 (no instruction) in the same batch equals its official row, padded to the longer length
n1 = official["attention_mask"][1].sum().item()
assert torch.equal(b["input_ids"][1, :n1], official["input_ids"][1, :n1]) and b["attention_mask"][1].sum().item() == n1
assert (b["attention_mask"][1, n1:] == 0).all()
print("3. mixed batch (one row with, one without instruction): ok")
