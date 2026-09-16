"""GPU-side data pipeline for Google Speech Commands.

All audio is held as one int16 tensor on the GPU; batching, time shift,
background-noise mixing, log-mel extraction and SpecAugment are vectorised,
so an epoch costs only the model's forward/backward time.

Protocols (matching the literature):
  v1_12 / v2_12 : 10 keywords + _unknown_ + _silence_, trained on the train
                  split, tested on the official 12-class test set
                  (as in BC-ResNet).
  v2_35         : all 35 words, standard testing_list split (as in KWT /
                  Keyword Mamba).
"""
import math

import torch
import torch.nn.functional as F
import torchaudio

SR = 16000
KEYWORDS10 = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
LABELS12 = ["_silence_", "_unknown_"] + KEYWORDS10
# Per-split size of the _unknown_ and _silence_ classes (train, valid) used by BC-ResNet.
BALANCE = {1: (1854, 258), 2: (3077, 371)}


class GSC:
    def __init__(self, task, cache_dir, device, seed=0):
        self.task, self.device = task, device
        ver = int(task[1])
        main = torch.load(f"{cache_dir}/gsc_v{ver}.pt")
        g = torch.Generator().manual_seed(seed)
        audio, label, split = main["audio"], main["label"].long(), main["split"].long()
        words = main["words"]

        noise = torch.load(f"{cache_dir}/noise.pt")
        # Pre-cut noise into overlapping 1 s windows for vectorised sampling.
        segs = []
        for n in noise:
            for s in range(0, len(n) - SR, SR // 10):
                segs.append(n[s : s + SR])
        self.noise = torch.stack(segs).to(device)

        if task.endswith("_35"):
            self.labels = words
            self.train_audio = audio[split == 0].to(device)
            self.train_label = label[split == 0].to(device)
            self.unknown_pool = None
            self.valid = (audio[split == 1], label[split == 1])
            self.test = (audio[split == 2], label[split == 2])
        else:
            self.labels = LABELS12
            n_bal_train, n_bal_valid = BALANCE[ver]
            word_to_12 = torch.tensor(
                [LABELS12.index(w) if w in KEYWORDS10 else 1 for w in words], dtype=torch.long
            )
            label12 = word_to_12[label]
            is_kw = label12 >= 2

            tr = split == 0
            self.train_audio = audio[tr & is_kw].to(device)
            self.train_label = label12[tr & is_kw].to(device)
            # _unknown_ is re-sampled from the whole unknown training pool every epoch.
            self.unknown_pool = audio[tr & ~is_kw].to(device)
            self.n_balance = n_bal_train

            va = split == 1
            unk_va = audio[va & ~is_kw]
            pick = torch.randperm(len(unk_va), generator=g)[:n_bal_valid]
            sil_va = self._silence(n_bal_valid, g)
            self.valid = (
                torch.cat([audio[va & is_kw], unk_va[pick], sil_va]),
                torch.cat(
                    [
                        label12[va & is_kw],
                        torch.ones(n_bal_valid, dtype=torch.long),
                        torch.zeros(n_bal_valid, dtype=torch.long),
                    ]
                ),
            )

            t = torch.load(f"{cache_dir}/gsc_v{ver}_test12.pt")
            self.test = (t["audio"], torch.tensor([LABELS12.index(c) for c in t["class_name"]]))

        self.num_classes = len(self.labels)

    def _silence(self, n, g):
        """Silence clips: background noise at a random level in [0, 1] (int16 scale)."""
        idx = torch.randint(len(self.noise), (n,), generator=g)
        amp = torch.rand(n, 1, generator=g)
        x = (self.noise.cpu()[idx] * amp).clamp(-1, 1)
        return (x * 32767).round().to(torch.int16)

    def train_epoch(self, batch_size, generator):
        """Yield (float waveform, label) GPU batches for one epoch."""
        audio, label = self.train_audio, self.train_label
        n_sil = 0
        if self.unknown_pool is not None:
            k = self.n_balance
            pick = torch.randperm(len(self.unknown_pool), device=self.device)[:k]
            audio = torch.cat([audio, self.unknown_pool[pick]])
            label = torch.cat([label, torch.ones(k, dtype=torch.long, device=self.device)])
            n_sil = k
        n_total = len(audio) + n_sil
        order = torch.randperm(n_total, device=self.device)
        for i in range(math.floor(n_total / batch_size)):
            idx = order[i * batch_size : (i + 1) * batch_size]
            is_sil = idx >= len(audio)
            safe = torch.where(is_sil, torch.zeros_like(idx), idx)
            x = audio[safe].float() / 32768.0
            x[is_sil] = 0.0
            y = torch.where(is_sil, torch.zeros_like(idx), label[safe])
            yield x, y


class Augment(torch.nn.Module):
    """Vectorised waveform augmentation: time shift + background noise (BC-ResNet recipe)."""

    def __init__(self, noise, max_shift=0.1, noise_prob=0.8, kw_noise_amp=0.1, has_silence=True):
        super().__init__()
        self.noise = noise
        self.has_silence = has_silence  # class 0 is _silence_ only in the 12-class tasks
        self.max_shift = int(max_shift * SR)
        self.noise_prob, self.kw_noise_amp = noise_prob, kw_noise_amp

    def forward(self, x, y, return_parts=False):
        b, n = x.shape
        dev = x.device
        # time shift with zero padding
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (b, 1), device=dev)
        pos = torch.arange(n, device=dev)[None] + shift
        valid = (pos >= 0) & (pos < n)
        x = torch.gather(x, 1, pos.clamp(0, n - 1)) * valid
        # noise: always for silence (amp U(0,1)), with prob p for others (amp U(0,0.1))
        is_sil = (y == 0) if self.has_silence else torch.zeros_like(y, dtype=torch.bool)
        amp = torch.rand(b, 1, device=dev)
        amp = torch.where(is_sil[:, None], amp, amp * self.kw_noise_amp)
        use = is_sil | (torch.rand(b, device=dev) < self.noise_prob)
        noise = self.noise[torch.randint(len(self.noise), (b,), device=dev)] * amp * use[:, None]
        mix = (x + noise).clamp(-1.0, 1.0)
        return (mix, x, noise) if return_parts else mix


def speech_presence_targets(clean, noise, win=480, hop=160, snr_scale=3.0):
    """Soft per-frame speech-presence labels in [0, 1], aligned with the log-mel frames.

    Free supervision: during augmentation we know the clean clip and the added noise
    separately, so the frame SNR is exact. A frame counts as speech when
      (a) clean energy exceeds the added noise energy        (frame SNR > 0 dB),
      (b) it is within 25 dB of the clip's loudest frame      (not a pause), and
      (c) it is above an absolute floor of -65 dBFS           (not digital silence).
    Each condition is a soft sigmoid; the target is their product.
    """
    def frame_energy(s):
        s = F.pad(s, (win // 2, win // 2))
        return s.unfold(1, win, hop).pow(2).mean(-1)  # (B, 1 + len/hop)

    es, en = frame_energy(clean), frame_energy(noise)
    db_s = 10 * torch.log10(es + 1e-12)
    snr = db_s - 10 * torch.log10(en + 1e-12)
    rel = db_s - db_s.max(dim=1, keepdim=True).values
    return (torch.sigmoid(snr / snr_scale) * torch.sigmoid((rel + 25) / snr_scale)
            * torch.sigmoid((db_s + 65) / snr_scale))


class LogMel(torch.nn.Module):
    """40-band log-mel, 30 ms window, 10 ms hop -> (B, 1, 40, 101). Same as BC-ResNet."""

    def __init__(self, n_mels=40, n_fft=512, win=480, hop=160):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, win_length=win, hop_length=hop, n_mels=n_mels
        )

    @torch.no_grad()
    def forward(self, x):
        return (self.mel(x.float()) + 1e-6).log().unsqueeze(1)


def spec_augment(x, f_para=7, t_para=20, f_num=2, t_num=2, return_time_keep=False):
    """Vectorised SpecAugment on (B, 1, F, T): zero random frequency and time bands."""
    b, _, nf, nt = x.shape
    dev = x.device
    keep = torch.ones(b, 1, nf, nt, device=dev, dtype=torch.bool)
    time_keep = torch.ones(b, nt, device=dev, dtype=torch.bool)
    for size, para, num, dim in ((nf, f_para, f_num, 2), (nt, t_para, t_num, 3)):
        for _ in range(num):
            w = (torch.rand(b, device=dev) * para).long()
            start = (torch.rand(b, device=dev) * (size - w + 1)).long()
            ar = torch.arange(size, device=dev)[None]
            m = (ar >= start[:, None]) & (ar < (start + w)[:, None])
            keep &= ~(m[:, None, :, None] if dim == 2 else m[:, None, None, :])
            if dim == 3:
                time_keep &= ~m
    return (x * keep, time_keep) if return_time_keep else x * keep


def eval_batches(audio, label, batch_size, device):
    for i in range(0, len(audio), batch_size):
        yield (
            audio[i : i + batch_size].to(device).float() / 32768.0,
            label[i : i + batch_size].to(device),
        )
