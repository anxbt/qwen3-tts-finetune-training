"""Multi-speaker version of QwenLM/Qwen3-TTS finetuning/sft_12hz.py.

Differences from the official script:
  - several named speakers (--speakers_json) instead of one, stored in embedding rows FIRST_ROW, FIRST_ROW+1, ...
  - speaker vector = mean over up to 50 clips per speaker, not the first batch's vector
  - local model path (the official copytree needs a folder), sdpa attention
  - text_projection applied and losses aligned with how generation works (see compute_loss)
  - resume (--resume_dir), with or without optimizer state; saves every N steps, keeps the 2 newest; budget stop (--max_usd)
  - validation loss at every save (--val_jsonl), appended to val_log.jsonl
  - instruction slot (row field "instruct", see instruct_dataset.py) and a per-row speaker position (batch["spk_pos"])
  - several GPUs: the loss runs inside one nn.Module (LossModule) so DistributedDDP's forward and gradient hooks are used;
    launch with `accelerate launch --num_processes N sft_speakers.py ...`. Validation, saving and printing run on the main process.
Run with `accelerate launch` on one or more GPUs. Uses Qwen's official dataset.py, vendored in third_party/qwen_finetuning
(override with QWEN_FT_DIR).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, send_to_device, set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

sys.path.insert(0, os.environ.get("QWEN_FT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "qwen_finetuning")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruct_dataset import InstructTTSDataset  # noqa: E402  (imports the official TTSDataset)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

FIRST_ROW = 3000  # speaker rows follow speakers.json order: FIRST_ROW, FIRST_ROW + 1, ...
VAL_BATCHES = 64  # --val_batches overrides: how many batches of each validation file are scored at a save


def newest_dirs(parent, prefix, keep, protect=()):
    """Delete all but the `keep` newest folders named <prefix><step>, except any name in `protect` (the best step)."""
    names = sorted((d for d in os.listdir(parent) if d.startswith(prefix)), key=lambda d: int(d.rsplit("-", 1)[1]))
    for d in names[:-keep]:
        if d not in protect:
            shutil.rmtree(os.path.join(parent, d))


def load_best(output_dir):
    """Best val_loss so far, kept in best-checkpoint/best.json, so a restart does not forget it."""
    try:
        return json.load(open(os.path.join(output_dir, "best-checkpoint", "best.json")))
    except (OSError, ValueError):
        return {"loss": float("inf"), "step": None}


def lr_at(step, total, warm, peak, floor):
    """Linear warmup over `warm` batches, then cosine from `peak` down to `floor * peak` at `total`.
    Computed from the batch count alone, so a resumed run continues the same curve without scheduler state."""
    if step < warm:
        return peak * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


assert lr_at(0, 100, 10, 1.0, 0.1) == 0.1 and lr_at(9, 100, 10, 1.0, 0.1) == 1.0  # warmup reaches peak
assert lr_at(10, 100, 10, 1.0, 0.1) == 1.0 and abs(lr_at(55, 100, 10, 1.0, 0.1) - 0.55) < 1e-9  # midpoint = halfway to floor
assert abs(lr_at(100, 100, 10, 1.0, 0.1) - 0.1) < 1e-9 and abs(lr_at(500, 100, 10, 1.0, 0.1) - 0.1) < 1e-9  # ends and stays at floor


def read_jsonl(path):
    """One dict per line. audio_codes become int16 numpy arrays (codec ids are below 2048): a 500 h set has about 250k rows, and
    Python lists of ints would need tens of GB of RAM per process (and per forked loader worker)."""
    import numpy as np

    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "audio_codes" in r:
            r["audio_codes"] = np.asarray(r["audio_codes"], dtype=np.int16)
        rows.append(r)
    return rows


def compute_loss(model, batch):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.device).to(model.dtype)).detach()

    # Changed from the official script: generation always applies text_projection to text embeddings
    # (modeling_qwen3_tts.py lines ~2079-2230), the official training loop does not.
    input_text_embedding = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
    rows = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
    input_codec_embedding[rows, batch["spk_pos"].to(input_codec_embedding.device), :] = speaker_embedding  # slot 6, or 6 + instruction length
    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        input_embeddings = input_embeddings + codec_i_embedding * codec_mask.unsqueeze(-1)

    # Changed from the official script. The talker classes use Hugging Face ForCausalLMLoss, which shifts
    # labels by one itself, and the official script also shifts by hand (a 2-frame-ahead target). So the
    # losses are computed here with explicit alignment: position p predicts codec 0 of position p+1.
    outputs = model.talker(inputs_embeds=input_embeddings[:, :-1, :], attention_mask=attention_mask[:, :-1], output_hidden_states=True)
    talker_loss = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]), codec_0_labels[:, 1:].reshape(-1), ignore_index=-100)
    hidden_states = outputs.hidden_states[0][-1]
    # The hidden state that produced frame f's codec 0 sits one position BEFORE frame f's input. The sub-talker
    # predicts codebooks 1..15 of frame f from it, so pair hidden(p-1) with the codes of the frame at p.
    before_frame = torch.zeros_like(codec_mask)
    before_frame[:, :-1] = codec_mask[:, 1:]
    talker_codec_ids = codec_ids[codec_mask]
    sub_talker_logits, _ = model.talker.forward_sub_talker_finetune(talker_codec_ids, hidden_states[before_frame[:, :-1]])
    sub_talker_loss = F.cross_entropy(sub_talker_logits.float().reshape(-1, sub_talker_logits.shape[-1]), talker_codec_ids[:, 1:].reshape(-1))
    return talker_loss + 0.3 * sub_talker_loss


class LossModule(torch.nn.Module):
    """Wraps the Qwen3-TTS model so the loss is computed inside forward(). DistributedDataParallel only syncs gradients when
    its own forward ran; calling model.talker(...) on the DDP wrapper (as compute_loss does) would skip that."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, batch):
        return compute_loss(self.inner, batch)


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model_path", required=True)  # local folder
    p.add_argument("--output_model_path", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", default="")
    p.add_argument("--val_emo_jsonl", default="")  # emotional clips only: logged as val_emo_loss next to val_loss
    p.add_argument("--speakers_json", required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)  # peak LR
    p.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")  # constant = old behaviour, for resuming runs started before this
    p.add_argument("--warmup_ratio", type=float, default=0.03)  # share of all batches spent warming up
    p.add_argument("--min_lr_ratio", type=float, default=0.1)  # cosine ends at this share of --lr
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=4)  # official value; per update = batch_size x grad_accum x number of GPUs
    p.add_argument("--workers", type=int, default=0)  # dataloader processes per GPU (mel spectrograms are built on the CPU)
    p.add_argument("--find_unused", action="store_true")  # DDP: only if it reports parameters that got no gradient
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_batches", type=int, default=VAL_BATCHES)
    p.add_argument("--max_steps", type=int, default=0)  # stop after this many batches (0 = no limit)
    p.add_argument("--save_every", type=int, default=0)  # batches between saves (0 = only at the end)
    p.add_argument("--resume_dir", default="")
    p.add_argument("--init_weights", default="")  # a folder with model.safetensors to warm-start from, step 0, no optimizer
    p.add_argument("--max_usd", type=float, default=0)
    p.add_argument("--usd_per_hour", type=float, default=0)
    p.add_argument("--spent_usd", type=float, default=0)  # already spent before this run
    p.add_argument("--no_save", action="store_true")  # timing/memory runs: skip the final checkpoint + optimizer save
    p.add_argument("--step_log", default="")  # timing runs: one JSON line per batch (wall seconds, audio seconds, peak memory)
    args = p.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16",
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused)])
    set_seed(args.seed)
    os.makedirs(args.output_model_path, exist_ok=True)  # a plain launch may not have made it
    MODEL_PATH = args.init_model_path
    qwen3tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    inner = qwen3tts.model
    inner.speaker_encoder.requires_grad_(False)  # frozen: its output is detached, so DDP must not wait for its gradients

    dataset = InstructTTSDataset(read_jsonl(args.train_jsonl), qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=args.workers,
                                  pin_memory=args.workers > 0, persistent_workers=args.workers > 0)
    val_loader = None
    if args.val_jsonl:
        val_set = InstructTTSDataset(read_jsonl(args.val_jsonl), qwen3tts.processor, config)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_set.collate_fn)
    val_emo_loader = None
    if args.val_emo_jsonl:
        val_emo_set = InstructTTSDataset(read_jsonl(args.val_emo_jsonl), qwen3tts.processor, config)
        val_emo_loader = DataLoader(val_emo_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_emo_set.collate_fn)
    optimizer = AdamW([q for q in inner.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(LossModule(inner), optimizer, train_dataloader)  # `inner` stays the plain model for attributes and saving

    speakers = json.load(open(args.speakers_json, encoding="utf-8"))
    for name in speakers:
        assert name == name.lower(), f"speaker name must be lowercase: {name}"

    start_epoch, start_step, spent = 0, 0, args.spent_usd
    named = {}
    if args.resume_dir:
        if os.path.exists(os.path.join(args.resume_dir, "optimizer.bin")):
            accelerator.load_state(args.resume_dir)
        else:  # weights only: the optimizer restarts fresh
            missing = inner.load_state_dict(load_file(os.path.join(args.resume_dir, "model.safetensors")), strict=False)
            accelerator.print(f"resumed from weights only, no optimizer state. missing keys: {len(missing.missing_keys)}")
        st = json.load(open(os.path.join(args.resume_dir, "trainer.json")))
        start_epoch, start_step, spent = st["epoch"], st["step"], st["spent_usd"]
        accelerator.print(f"resumed at epoch {start_epoch} step {start_step}, spent ${spent:.2f}")
        saved_named = os.path.join(args.resume_dir, "named.json")
        if os.path.exists(saved_named):  # the voice clips may not exist on another account
            named = {n: torch.tensor(v, dtype=inner.dtype, device=inner.device) for n, v in json.load(open(saved_named)).items()}
    elif args.init_weights:
        missing = inner.load_state_dict(load_file(os.path.join(args.init_weights, "model.safetensors")), strict=False)
        accelerator.print(f"warm start from {args.init_weights}, missing keys: {len(missing.missing_keys)}")

    # One averaged speaker vector per named speaker (encoder is frozen, so once is enough).
    if not named:
        with torch.no_grad():
            for name, paths in speakers.items():
                embs = []
                for path in paths[:50]:
                    wav, sr = dataset._load_audio_to_np(path)
                    mel = dataset.extract_mels(audio=wav, sr=sr)
                    embs.append(inner.speaker_encoder(mel.to(inner.device).to(inner.dtype)).detach())
                named[name] = torch.cat(embs).mean(0)
    accelerator.print({n: f"{v.float().norm().item():.2f}" for n, v in named.items()}, "speaker vector norms")

    run_start, global_step, epoch = time.time(), start_step, start_epoch
    last_saved = {"step": -1}
    best = load_best(args.output_model_path)

    def spent_now():
        return spent + (time.time() - run_start) / 3600 * args.usd_per_hour

    def run_val(loader):
        total, n = 0.0, 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.val_batches:
                    break
                total += compute_loss(inner, send_to_device(batch, inner.device)).item()
                n += 1
        model.train()
        return total / max(n, 1), n

    def validate():
        if val_loader is None or not accelerator.is_main_process:  # only the main process validates; the others wait at the next sync
            return None
        val_loss, n = run_val(val_loader)
        accelerator.print(f"VAL | Step {global_step} | val_loss {val_loss:.4f} over {n} batches")
        entry = {"step": global_step, "val_loss": val_loss, "spent_usd": spent_now()}
        if val_emo_loader is not None:
            entry["val_emo_loss"], n_emo = run_val(val_emo_loader)
            accelerator.print(f"VAL | Step {global_step} | val_emo_loss {entry['val_emo_loss']:.4f} over {n_emo} batches")
        with open(os.path.join(args.output_model_path, "val_log.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        return val_loss

    def save_all(tag):
        """Validation, resumable state and a usable checkpoint. COMMIT is written last, so a partial save is never used."""
        val_loss = validate()  # None on non-main processes
        last_saved["step"] = global_step
        main = accelerator.is_main_process
        output_dir = os.path.join(args.output_model_path, f"checkpoint-{tag}")
        if main:
            write_checkpoint(output_dir)
        rdir = os.path.join(args.output_model_path, f"resume-{tag}")
        accelerator.save_state(rdir)  # every process calls it; accelerate writes the files once
        if main:
            finish_save(tag, rdir, output_dir, val_loss)
        accelerator.wait_for_everyone()

    def write_checkpoint(output_dir):
        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
        cfg = json.load(open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8"))
        cfg["tts_model_type"] = "custom_voice"
        t = cfg.get("talker_config", {})
        t["spk_id"] = {n: FIRST_ROW + i for i, n in enumerate(named)}
        t["spk_is_dialect"] = {n: False for n in named}
        cfg["talker_config"] = t
        json.dump(cfg, open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        state_dict = {k: v.detach().to("cpu") for k, v in inner.state_dict().items()}
        for k in [k for k in state_dict if k.startswith("speaker_encoder")]:
            del state_dict[k]
        weight = state_dict["talker.model.codec_embedding.weight"]
        for i, vec in enumerate(named.values()):
            weight[FIRST_ROW + i] = vec.detach().to("cpu").to(weight.dtype)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    def finish_save(tag, rdir, output_dir, val_loss):
        json.dump({"epoch": epoch, "step": global_step, "spent_usd": spent_now()}, open(os.path.join(rdir, "trainer.json"), "w"))
        json.dump({n: v.float().cpu().tolist() for n, v in named.items()}, open(os.path.join(rdir, "named.json"), "w"))
        open(os.path.join(rdir, "COMMIT"), "w").close()
        if val_loss is not None and val_loss < best["loss"]:
            best["loss"], best["step"] = val_loss, tag
            best_dir = os.path.join(args.output_model_path, "best-checkpoint")
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(output_dir, best_dir)
            json.dump(best, open(os.path.join(best_dir, "best.json"), "w"))
            accelerator.print(f"new best checkpoint: step {tag}, val_loss {val_loss:.4f}")

        # Prune after the best is known: the 2 newest of each, plus the best step's folders (it used to be deleted once 2 newer saves existed).
        keep_best = {f"resume-{best['step']}", f"checkpoint-{best['step']}"} if best["step"] is not None else set()
        newest_dirs(args.output_model_path, "resume-", 2, keep_best)
        newest_dirs(args.output_model_path, "checkpoint-", 2, keep_best)
        accelerator.print(f"saved resume-{tag} and checkpoint-{tag}")

    if global_step == 0:  # baseline: how the warm-started (or fresh) model scores before any new-data training
        validate()

    state, skipped = "done", 0
    status_file = os.path.join(args.output_model_path, "STATUS")
    if accelerator.is_main_process and os.path.exists(status_file):  # left by an earlier stop: the Mac sync must not take this run as finished
        os.remove(status_file)
    model.train()
    stop = False
    steps_per_epoch = max(len(train_dataloader), 1)
    total_steps = args.num_epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm_steps = max(1, int(total_steps * args.warmup_ratio))
    accelerator.print(f"LR schedule: {args.schedule}, peak {args.lr}, warmup {warm_steps} of {total_steps} batches, floor {args.min_lr_ratio * args.lr:.2e}")
    step_log = open(args.step_log, "w") if args.step_log and accelerator.is_main_process else None
    t_prev = time.time()
    start_epoch = start_step // steps_per_epoch  # a save on an epoch's last batch resumes at the next epoch, not the same one again
    for epoch in range(start_epoch, args.num_epochs):
        skip = start_step % steps_per_epoch if epoch == start_epoch else 0
        loader = accelerator.skip_first_batches(train_dataloader, skip) if skip else train_dataloader
        for batch in loader:
            oom = False
            if args.schedule == "cosine":
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(global_step, total_steps, warm_steps, args.lr, args.min_lr_ratio)
            try:
                with accelerator.accumulate(model):
                    loss = model(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            except torch.OutOfMemoryError:
                if accelerator.num_processes > 1:
                    raise  # a skipped batch on one GPU would leave the others waiting in the gradient all-reduce
                oom = True
            if step_log:  # synchronize so the wall time is the batch's real GPU time (data loading included)
                torch.cuda.synchronize()
                now = time.time()
                step_log.write(json.dumps({"step": global_step + (not oom), "secs": now - t_prev, "oom": oom, "sync": bool(accelerator.sync_gradients),
                                           "audio_s": batch["codec_mask"].sum().item() / 12, "clips": batch["codec_ids"].shape[0],
                                           "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}) + "\n")
                step_log.flush()
                t_prev = now
            if oom:  # a batch of long clips can exceed the card: skip it. Handled after the except block, which still holds the batch's tensors
                skipped += 1
                accelerator.print(f"OOM at step {global_step + 1}, batch skipped ({skipped} so far)")
                if skipped > 100:
                    raise RuntimeError("more than 100 batches ran out of memory: lower the batch size")
                loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                mem = torch.cuda.max_memory_allocated() / 1e9
                accelerator.print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | peak GPU {mem:.1f} GB (reserved {torch.cuda.max_memory_reserved() / 1e9:.1f}) | elapsed {time.time() - run_start:.0f}s | spent ${spent_now():.2f}")
            if args.save_every and global_step % args.save_every == 0:
                save_all(global_step)
            if args.max_usd:
                over = torch.tensor(float(spent_now() >= args.max_usd), device=accelerator.device)
                if accelerator.num_processes > 1:
                    over = accelerator.reduce(over, "max")  # every GPU must take the same decision, or one waits forever
                if over.item() >= 1:
                    accelerator.print("budget reached, saving and stopping")
                    state, stop = "budget", True
            if args.max_steps and global_step >= args.max_steps:
                state, stop = "max_steps", True
            if stop:
                break
        if stop:
            break

    if not args.no_save and last_saved["step"] != global_step:  # the final step is always saved; --save_every only adds saves in between
        save_all(global_step)
    if accelerator.is_main_process:
        json.dump({"state": state, "step": global_step, "spent_usd": spent_now(), "oom_skipped": skipped}, open(status_file, "w"))
    accelerator.wait_for_everyone()
    accelerator.print(f"finished: {state}, {skipped} batches skipped for OOM")


if __name__ == "__main__":
    train()
