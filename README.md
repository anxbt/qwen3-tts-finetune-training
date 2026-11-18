# Qwen3-TTS fine-tuning: named speakers + instruction slot

Training code for fine-tuning [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) on new languages and voices,
with style control through Qwen's **instruction slot** (an English sentence such as `Speak in a sad tone, at a slow pace.` that steers delivery
and is never spoken). It extends Qwen's official [`finetuning/sft_12hz.py`](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) recipe.

One trainer, reused for every run; runs differ only in data and command-line flags.

## Where the recipe comes from

All links are pinned to Qwen3-TTS commit `022e286b98fbec7e1e916cb940cdf532cd9f488e` (the commit `third_party/` was copied from), so the line numbers stay valid.
Official script: [`finetuning/sft_12hz.py`](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py)

| Setting | Official value | Line |
|---|---|---|
| batch size per GPU | 2 | [L38](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L38) |
| learning rate | 2e-5 | [L39](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L39) |
| epochs | 3 | [L40](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L40) |
| gradient accumulation 4, bf16 | `Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")` | [L44](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L44) |
| optimizer | AdamW, weight decay 0.01, constant lr (no scheduler) | [L60](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L60) |
| speaker vector at slot 6 | `input_codec_embedding[:, 6, :] = speaker_embedding` | [L91](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L91) |
| loss | `outputs.loss + 0.3 * sub_talker_loss` | [L113](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L113) |
| gradient clip | 1.0 | [L118](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L118) |
| saved speaker row | 3000 | [L156](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L156) |

Where this code deliberately differs from that script (details in `REVIEW.md`):

- text embedding without `text_projection` in the official loop: [L89](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L89); generation applies it, so we do too.
- labels passed pre-shifted to a loss that shifts again: [L103](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py#L103); we align explicitly.

Inference layout we match: high-level [`generate_custom_voice`, `non_streaming_mode=True` (L738)](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py#L738),
which builds the whole-text-first prompt in [`generate` (L2022)](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py#L2022)
([non-streaming branch, L2203](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py#L2203)).

## Files

| File | What it does |
|---|---|
| `train/sft_speakers.py` | Trainer: loads the base model, builds speaker vectors, computes the loss, runs single- or multi-GPU training with `accelerate`, validates, saves and resumes. The top docstring lists every difference from the official script. |
| `train/instruct_dataset.py` | Dataset + collate: Qwen's official layout plus the instruction block in front of it. |
| `train/test_instruct_dataset.py` | CPU-only check of the collate layout (no GPU, no model download). |
| `third_party/qwen_finetuning/dataset.py` | Qwen's official `TTSDataset`, copied unchanged (see the NOTICE there). |
| `REVIEW.md` | Reading order and the decisions worth checking. |

Data preparation (filtering, splits, writing audio, speech-code encoding) and inference are not in this repo; the trainer only needs the files below.

## Inputs

**Train / validation file** (JSON lines), one clip per line:

```json
{"text": "मेरे दादाजी अब हमारे बीच नहीं रहे।",
 "instruct": "Speak in a sad tone, at a slow pace.",
 "audio_codes": [[1532, 87, 902, ...], ...],
 "ref_audio": "refs/voice_ref_1.wav"}
```

- `text`: words to speak.
- `instruct`: optional style sentence; empty or missing = no instruction (the official layout).
- `audio_codes`: the clip encoded with `Qwen3-TTS-Tokenizer-12Hz`: one list of 16 codebook ids per frame, 12.5 frames per second.
- `ref_audio`: a reference clip of the same voice, 24 kHz mono, used for the speaker encoder. All reference clips must have the same length (the official collate concatenates them); the runs used exactly 6.0 s.
- Other fields (for example `audio`, `language`, `voice`) are ignored by the trainer.

**Speakers file** (`--speakers_json`): `{"speaker_name": ["clip1.wav", "clip2.wav", ...], ...}`. Names must be lowercase. Each speaker's vector is the
mean speaker-encoder output over up to 50 of its clips; vectors are written to codec-embedding rows 3000, 3001, ... in file order and registered in
the saved `config.json` (`talker_config.spk_id`), so the checkpoint loads as a `custom_voice` model with those speaker names.

## Run

```bash
pip install -r requirements.txt
python train/test_instruct_dataset.py        # layout check, CPU, a few seconds

accelerate launch --num_processes 4 --mixed_precision bf16 train/sft_speakers.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path out/run1 \
  --train_jsonl train_with_codes.jsonl --val_jsonl val_with_codes.jsonl --speakers_json speakers.json \
  --batch_size 2 --grad_accum 4 --lr 2e-5 --schedule constant --num_epochs 3 \
  --save_every 7250 --val_batches 300 --workers 6
```

Clips per optimizer update = `batch_size x grad_accum x GPUs` (32 above). For one GPU use `--num_processes 1` or plain `python`.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--init_model_path` | required | local folder of the base model |
| `--output_model_path` | required | where checkpoints and logs go |
| `--train_jsonl` | required | training rows |
| `--val_jsonl` | none | validation rows; scored at every save |
| `--val_emo_jsonl` | none | optional second validation file, logged as `val_emo_loss` |
| `--speakers_json` | required | speaker name -> clips |
| `--batch_size` | 2 | clips per GPU per step (official value) |
| `--grad_accum` | 4 | steps per optimizer update (official value) |
| `--lr` | 2e-5 | peak learning rate (official value) |
| `--schedule` | cosine | `cosine` (linear warmup, then cosine down to `min_lr_ratio x lr`) or `constant` (flat `--lr` from step 1, no warmup; the official recipe) |
| `--warmup_ratio` | 0.03 | share of all steps spent warming up (`cosine` only; the startup log prints a warmup count for `constant` too, but it is not applied) |
| `--min_lr_ratio` | 0.1 | cosine floor as a share of `--lr` |
| `--num_epochs` | 3 | epochs |
| `--save_every` | 0 | steps between save + validation (0 = only at the end) |
| `--val_batches` | 64 | batches of each validation file scored per save |
| `--workers` | 0 | dataloader processes per GPU (mel features are built on the CPU) |
| `--seed` | 1234 | random seed |
| `--resume_dir` | none | resume from a `resume-<step>` folder (with or without optimizer state) |
| `--init_weights` | none | warm-start weights from a folder, step 0, fresh optimizer |
| `--max_steps` | 0 | stop after this many steps (0 = no limit) |
| `--find_unused` | off | DDP option, only if it reports parameters without gradients |
| `--max_usd`, `--usd_per_hour`, `--spent_usd` | 0 | optional budget stop for hourly-billed GPUs |
| `--no_save`, `--step_log` | off | timing runs: skip saving / log per-step timing |

Fixed in code: AdamW (weight decay 0.01), gradient clip 1.0, bf16, sdpa attention, frozen speaker encoder,
loss = talker cross-entropy on codebook 0 + 0.3 x sub-talker cross-entropy on codebooks 1-15.

## Outputs (in `--output_model_path`)

- `checkpoint-<step>/`: loadable model folder (weights, config with speaker names, tokenizer files).
- `resume-<step>/`: optimizer and trainer state; `COMMIT` is written last, so a partial save is never resumed from.
- `best-checkpoint/` + `best.json`: lowest validation loss so far.
- `val_log.jsonl`: one line per save (`step`, `val_loss`, optional `val_emo_loss`).
- `STATUS`: final state. The 2 newest `checkpoint-*` / `resume-*` folders are kept, plus the best step's.

## Status

This code trained three runs (Hindi + Kannada emotional speech with 4 and with 2 voices; 500 h of synthetic Hindi / Odia / Tamil / Kannada with one voice).
They were judged by validation loss only; no listening test, CER, WER or MOS has been run yet. Datasets and checkpoints are private.
Known limit: clips were 2-25 s, and a single generation over a much longer text tends to stop early, so long text should be split into sentences at inference.

## License

Apache-2.0 (`LICENSE`). `third_party/qwen_finetuning/dataset.py` is Copyright the Alibaba Qwen team, Apache-2.0.
