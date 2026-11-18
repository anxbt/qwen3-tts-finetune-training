"""TTSDataset (official finetuning/dataset.py) plus Qwen3-TTS's instruction slot.

Row fields: audio_codes, text, ref_audio (as the official dataset), plus optional `instruct` (a plain sentence such as "Speak sadly.").
Layout per row (this is the order generation uses, qwen_tts modeling: instruct embedding first, then the role tokens):

    [instruct: <|im_start|>user\n{instruct}<|im_end|>\n][role][codec prefix + speaker][text][audio frames]

The instruction positions carry a text embedding only (no codec embedding), the label is -100 there, and every
later position of the official layout shifts right by L = instruction length. With an empty instruction L = 0 and the
batch equals the official collate_fn output, plus `spk_pos` (the speaker slot, 6 + L, per row).
"""
import torch
from dataset import TTSDataset  # third_party/qwen_finetuning/dataset.py, put on sys.path by the caller


class InstructTTSDataset(TTSDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        instruct = (self.data_list[idx].get("instruct") or "").strip()
        if instruct:
            ids = self._tokenize_texts(f"<|im_start|>user\n{instruct}<|im_end|>\n")
            item["instruct_ids"] = ids[:, :]  # 1, L
        else:
            item["instruct_ids"] = torch.zeros((1, 0), dtype=torch.long)
        return item

    def collate_fn(self, batch):
        assert self.lag_num == -1
        cfg, tcfg = self.config, self.config.talker_config
        lens = [b["instruct_ids"].shape[1] for b in batch]
        item_length = [L + b["text_ids"].shape[1] + b["audio_codes"].shape[0] for L, b in zip(lens, batch)]
        b, t = len(batch), max(item_length) + 8

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
        spk_pos = torch.zeros(b, dtype=torch.long)

        for i, data in enumerate(batch):
            o = lens[i]  # offset: everything of the official layout moves right by the instruction length
            text_ids, audio_codecs = data["text_ids"], data["audio_codes"]
            audio_codec_0 = audio_codecs[:, 0]
            n_text, n_codec = text_ids.shape[1], audio_codec_0.shape[0]

            # instruction: text channel only
            input_ids[i, :o, 0] = data["instruct_ids"][0]
            # text channel
            input_ids[i, o:o + 3, 0] = text_ids[0, :3]
            input_ids[i, o + 3:o + 7, 0] = cfg.tts_pad_token_id
            input_ids[i, o + 7, 0] = cfg.tts_bos_token_id
            input_ids[i, o + 8:o + 8 + n_text - 3, 0] = text_ids[0, 3:]
            input_ids[i, o + 8 + n_text - 3, 0] = cfg.tts_eos_token_id
            input_ids[i, o + 8 + n_text - 2:o + 8 + n_text + n_codec, 0] = cfg.tts_pad_token_id
            text_embedding_mask[i, :o + 8 + n_text + n_codec] = True

            # codec channel
            input_ids[i, o + 3:o + 8, 1] = torch.tensor([tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id, 0, tcfg.codec_pad_id])
            input_ids[i, o + 8:o + 8 + n_text - 3, 1] = tcfg.codec_pad_id
            input_ids[i, o + 8 + n_text - 3, 1] = tcfg.codec_pad_id
            input_ids[i, o + 8 + n_text - 2, 1] = tcfg.codec_bos_id
            input_ids[i, o + 8 + n_text - 1:o + 8 + n_text - 1 + n_codec, 1] = audio_codec_0
            input_ids[i, o + 8 + n_text - 1 + n_codec, 1] = tcfg.codec_eos_token_id

            codec_0_labels[i, o + 8 + n_text - 1:o + 8 + n_text - 1 + n_codec] = audio_codec_0
            codec_0_labels[i, o + 8 + n_text - 1 + n_codec] = tcfg.codec_eos_token_id
            codec_ids[i, o + 8 + n_text - 1:o + 8 + n_text - 1 + n_codec, :] = audio_codecs

            codec_embedding_mask[i, o + 3:o + 8 + n_text + n_codec] = True
            codec_embedding_mask[i, o + 6] = False  # speaker slot: filled with the speaker vector in compute_loss
            codec_mask[i, o + 8 + n_text - 1:o + 8 + n_text - 1 + n_codec] = True
            attention_mask[i, :o + 8 + n_text + n_codec] = True
            spk_pos[i] = o + 6

        return {
            "input_ids": input_ids, "ref_mels": torch.cat([d["ref_mel"] for d in batch], dim=0), "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1), "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels, "codec_ids": codec_ids, "codec_mask": codec_mask, "spk_pos": spk_pos,
        }
