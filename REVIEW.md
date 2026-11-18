# Review guide

Suggested reading order, and the decisions most worth checking. Line numbers refer to `train/sft_speakers.py` unless noted.

## 1. Sequence layout with the instruction slot (`train/instruct_dataset.py`)

The official collate (`third_party/qwen_finetuning/dataset.py`, `collate_fn`) builds two channels per position (text ids, codec ids):
role tokens, codec prefix with the speaker slot at position 6, the text, then the audio codes. `InstructTTSDataset` prepends the instruction,
tokenized as `<|im_start|>user\n{instruct}<|im_end|>\n`, and shifts the whole official layout right by its length L:

- instruction positions: text channel only (`text_embedding_mask` true, `codec_embedding_mask` false), labels -100 (never predicted);
- the speaker slot moves from 6 to 6 + L, returned per row as `spk_pos`;
- with an empty instruction, L = 0 and the batch equals the official one (`test_instruct_dataset.py` check 1).

Check: this matches Qwen's non-streaming inference prompt (instruction block first, then role tokens, codec prefix + speaker, text, audio),
as built in `modeling_qwen3_tts.py` `generate(..., non_streaming_mode=True)`, the default of the high-level `generate_custom_voice`.

## 2. Loss (`compute_loss`, lines ~85-122)

Three intentional differences from the official `sft_12hz.py`, each commented in the code:

1. `text_projection` is applied to the text embeddings, because generation applies it and the official training loop does not.
2. Labels are aligned explicitly (position p predicts codebook 0 of position p+1). The official script shifts by hand and the Hugging Face loss shifts again, a two-frame-ahead target.
3. The sub-talker (codebooks 1-15 of frame f) is fed the hidden state that produced frame f's codebook 0, i.e. one position before frame f.

Loss = talker cross-entropy (codebook 0) + 0.3 x sub-talker cross-entropy (codebooks 1-15).

## 3. Speaker vectors

- Training: each row's speaker vector is the frozen speaker encoder's output on that row's `ref_audio`, written into `spk_pos` (detached).
- Saved model: each named speaker's vector is the mean over up to 50 of its clips, stored in codec-embedding rows 3000+ and registered in `config.json`.
- So training sees a per-clip vector and inference a fixed mean vector. A known mismatch; its effect was not measured.

## 4. Training loop and multi-GPU

- `LossModule` wraps the loss so DDP's forward and gradient hooks run; the speaker encoder is frozen (`requires_grad_(False)`).
- `accelerator.accumulate`: gradients sync once per optimizer update; clip 1.0; AdamW weight decay 0.01; bf16.
- Validation, saving and printing on the main process only; all ranks call `save_state`; a barrier follows each save.
- Save is resumable: `resume-<step>` gets a `COMMIT` file last; resume picks only committed folders; the best checkpoint is never pruned.
- An out-of-memory batch is skipped on one GPU but re-raised with several GPUs (a skipped batch would stall the others in the all-reduce).

## 5. Schedule (worth knowing)

`--schedule constant` uses a flat learning rate from step 1 with **no warmup**, as in the official recipe. Warmup and decay apply only to `cosine`.
The startup line `LR schedule: constant, ..., warmup N of M batches` is misleading: it prints a warmup count that the constant schedule does not use.

## 6. What ran

This exact code (paths made repo-relative, comments tidied) trained three runs: two on ~21 h of Hindi + Kannada emotional speech (4 voices; 2 voices),
and one on 500 h of synthetic Hindi / Odia / Tamil / Kannada with one voice (4 GPUs). The two smaller runs used an earlier `sft_speakers.py` that differs
only in that `audio_codes` were kept as Python lists (now int16 arrays, a memory fix for the 500 h set) and validation batches were fixed at 64
(now `--val_batches`, default 64). Training math is the same.

Observed: the two ~21 h runs overfit after epoch 2 (validation loss rose while training loss kept falling); the 500 h run did not.
Evaluation so far is validation loss only: no listening test, CER, WER or MOS. At inference, one generation over text much longer than the
2-25 s training clips tends to stop early; splitting text into sentences avoids it.

## Questions a reviewer could answer

- Is the label alignment in `compute_loss` (items 2.2 and 2.3) right for Qwen3-TTS's talker / sub-talker split?
- Is it sound to train the instruction slot with instruction positions carrying text embeddings only and no codec embedding?
- Would averaging several reference clips per row in training (to match the stored mean vector) be better?
