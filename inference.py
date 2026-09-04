import math
import pathlib

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


sr = 16000
hop = 160
hop_time = hop / sr
win_frames = 64

fmin = 32.7
n_bins = 120
bins_per_octave = 24
fmax = fmin * (2 ** (n_bins / bins_per_octave))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cqt_freqs = fmin * (2 ** (np.arange(n_bins) / bins_per_octave))

conf_th_calib = 0.05
tempo_tol_calib = 10
onset_ratio_calib = 0.60

mpath = pathlib.Path(__file__).parent / "music.pt"
model = None


def hz_to_cents(hz):
    return 1200 * np.log2(hz / 32.7)


def cents_to_hz(c):
    return 32.7 * (2 ** (c / 1200))


def hz_to_bin(hz):
    if hz <= 0:
        return n_bins
    step = 1200 / bins_per_octave
    return int(np.clip(round(hz_to_cents(hz) / step), 0, n_bins - 1))


def bin_to_hz(b):
    return float(cqt_freqs[int(b)])


def hz_to_midi(hz):
    return 69 + 12 * math.log2(hz / 440.0)


def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12))


def cents_err(a, b):
    if a > 0 and b > 0:
        return 1200 * abs(math.log2(a / b))
    return 1e9


def compute_cqt(y):
    c = librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    m = np.abs(c).astype(np.float32)
    return np.log1p(m)


class MusicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(120, 64, 8, stride=2, padding=4)
        self.b1 = nn.BatchNorm1d(64)
        self.c2 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.b2 = nn.BatchNorm1d(128)
        self.c3 = nn.Conv1d(128, 256, 8, stride=2, padding=4)
        self.b3 = nn.BatchNorm1d(256)
        self.c4 = nn.Conv1d(256, 256, 4, stride=2, padding=2)
        self.b4 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, n_bins + 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = f.relu(self.b1(self.c1(x)))
        x = f.relu(self.b2(self.c2(x)))
        x = f.relu(self.b3(self.c3(x)))
        x = f.relu(self.b4(self.c4(x)))
        x = x.mean(dim=2)
        x = self.drop(f.relu(self.fc1(x)))
        return self.fc2(x)


def get_model():
    global model

    if model is not None:
        return model

    model = MusicModel().to(device)

    if mpath.exists():
        ckpt = torch.load(str(mpath), map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt)
        else:
            state = ckpt
        model.load_state_dict(state)

    model.eval()
    return model


def predict_frames(y, batch_size=256):
    m = get_model()
    c = compute_cqt(y)
    n = c.shape[1]

    out_hz = np.zeros(n, dtype=np.float32)
    out_conf = np.zeros(n, dtype=np.float32)
    out_bin = np.zeros(n, dtype=np.int32)
    half = win_frames // 2

    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            wins = []

            for i in range(s, e):
                ws = i - half
                we = ws + win_frames
                pl = max(0, -ws)
                pr = max(0, we - n)
                cs = max(0, ws)
                ce = min(n, we)

                w = c[:, cs:ce].T

                if pl > 0:
                    w = np.concatenate([np.zeros((pl, 120), dtype=np.float32), w])
                if pr > 0:
                    w = np.concatenate([w, np.zeros((pr, 120), dtype=np.float32)])

                wins.append(w)

            b = np.stack(wins)
            b = torch.from_numpy(b.transpose(0, 2, 1)).to(device)
            prob = torch.softmax(m(b), dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)

            for j, pb in enumerate(pred):
                idx = s + j

                if pb >= n_bins:
                    out_hz[idx] = 0
                    out_conf[idx] = float(prob[j, pb])
                    out_bin[idx] = n_bins
                else:
                    lo = max(0, int(pb) - 2)
                    hi = min(n_bins - 1, int(pb) + 2)
                    wp = prob[j, lo:hi + 1]
                    bi = np.arange(lo, hi + 1)
                    out_hz[idx] = float(np.sum(wp * cqt_freqs[bi]) / (np.sum(wp) + 1e-9))
                    out_conf[idx] = float(prob[j, pb])
                    out_bin[idx] = int(pb)

    times = np.arange(n, dtype=np.float32) * hop_time
    return times, out_hz, out_conf, out_bin


def load_audio(path, sr_in=sr):
    y, _ = librosa.load(str(path), sr=sr_in, mono=True)
    return y.astype(np.float32)


def detect_onsets(y, sr_in=sr):
    o = librosa.onset.onset_strength(y=y, sr=sr_in, hop_length=hop)
    t = librosa.onset.onset_detect(onset_envelope=o, sr=sr_in, hop_length=hop, units="time")
    return np.array(t, dtype=np.float32)


def estimate_tempo(y, sr_in=sr):
    o = librosa.onset.onset_strength(y=y, sr=sr_in, hop_length=hop)
    t = librosa.feature.tempo(onset_envelope=o, sr=sr_in, hop_length=hop)
    if len(t) > 0:
        return float(t[0])
    return 0.0


def dtw_align(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0), np.zeros(0)

    _, wp = librosa.sequence.dtw(X=a.reshape(1, -1), Y=b.reshape(1, -1), metric="euclidean")
    wp = np.array(wp)[::-1]
    return wp[:, 0], wp[:, 1]


def practice_report(ref_path, rec_path, conf_th=conf_th_calib, tempo_tol=tempo_tol_calib, ratio_tol=onset_ratio_calib):
    yref = load_audio(ref_path, sr)
    yrec = load_audio(rec_path, sr)

    tr, hr, cr, _ = predict_frames(yref)
    tp, hp, cp, _ = predict_frames(yrec)

    oref = detect_onsets(yref, sr)
    orec = detect_onsets(yrec, sr)

    tref = estimate_tempo(yref, sr)
    trec = estimate_tempo(yrec, sr)

    cref = librosa.feature.chroma_cqt(y=yref, sr=sr, hop_length=hop)
    crec = librosa.feature.chroma_cqt(y=yrec, sr=sr, hop_length=hop)

    cents = []
    tune = 0
    vo = 0

    if cref.shape[1] > 0 and crec.shape[1] > 0:
        _, wp = librosa.sequence.dtw(X=cref, Y=crec, metric="euclidean")
        wp = np.array(wp)[::-1]

        for r, c in wp:
            if r >= len(hr) or c >= len(hp):
                continue

            gr = float(hr[int(r)])
            pr = float(hp[int(c)])

            if gr < fmin or pr < fmin:
                continue
            if cr[int(r)] < conf_th or cp[int(c)] < conf_th:
                continue

            vo += 1
            cents.append(cents_err(pr, gr))

    if cents:
        octs = [min(v % 1200, 1200 - v % 1200) for v in cents]
        oacc = float(sum(1 for v in octs if v < 50) / max(len(octs), 1))
    else:
        octs = []
        oacc = 0.0

    for ce in cents:
        tune += int(ce < 50)

    pacc = tune / max(vo, 1)
    oscore = int(round(oacc * 100))

    if octs:
        moct = float(np.median(octs))
    else:
        moct = 0.0

    if cents:
        mcen = float(np.median(cents))
        meanc = float(np.mean(cents))
    else:
        mcen = 0.0
        meanc = 0.0

    err = []

    if len(oref) > 0 and len(orec) > 0:
        nref = oref - oref[0]
        nrec = orec - orec[0]
        ii, jj = dtw_align(nref, nrec)

        for k in range(len(ii)):
            err.append(abs(float(nref[int(ii[k])]) - float(nrec[int(jj[k])])) * 1000)

    if err:
        mon = float(np.median(err))
        meano = float(np.mean(err))
        oacc2 = float(np.mean(np.array(err) < 70))
    else:
        mon = 0.0
        meano = 0.0
        oacc2 = 0.0

    pscore = int(round(pacc * 100))
    tscore = int(round(oacc2 * 100))

    if tref > 0:
        tdiff = abs(tref - trec)
    else:
        tdiff = 0.0

    oratio = min(len(oref), len(orec)) / max(max(len(oref), len(orec)), 1)
    rel = (tdiff <= tempo_tol) and (oratio >= ratio_tol)

    if rel:
        trep = tscore
    else:
        trep = tscore // 2

    overall = int(round(0.3 * pscore + 0.3 * oscore + 0.4 * trep))

    issues = []

    if oacc < 0.75:
        issues.append(f"Octave low: {oscore}/100 | Median: {moct:.0f} cents")
    elif pacc < 0.75:
        issues.append(f"Pitch low: {pscore}/100 | Octave: {oscore}/100 | Median: {mcen:.0f} cents")
    elif mcen > 30:
        issues.append(f"Tuning drift: {mcen:.0f} cents")
    else:
        issues.append("Pitch solid")

    if not rel:
        issues.append(f"Timing unreliable: tempo {tref:.0f} vs {trec:.0f} bpm | onset {len(oref)} vs {len(orec)}")
    elif oacc2 < 0.7:
        issues.append(f"Timing off: median {mon:.0f} ms")
    else:
        issues.append("Timing solid")

    if tdiff > tempo_tol and tref > 0:
        issues.append(f"Tempo mismatch: ref {tref:.0f} vs rec {trec:.0f} bpm")

    rep = {
        "Pitch Score": pscore,
        "Octave Score": oscore,
        "Timing Score": trep,
        "Timing Reliable": rel,
        "Overall": overall,
        "Median Cents": mcen,
        "Median Octave": moct,
        "Mean Cents": meanc,
        "Median Onset (ms)": mon,
        "Mean Onset (ms)": meano,
        "Tempo Ref": tref,
        "Tempo Rec": trec,
        "Issues": issues,
        "Voiced Frames": vo,
        "Onsets Ref": len(oref),
        "Onsets Rec": len(orec),
    }

    return rep, (tr, hr, cr), (tp, hp, cp)


def predict_single_window(w):
    m = get_model()

    if w.shape[0] == win_frames:
        x = torch.from_numpy(w.T).unsqueeze(0).to(device).float()
    else:
        x = torch.from_numpy(w).unsqueeze(0).to(device).float()

    with torch.no_grad():
        prob = torch.softmax(m(x), dim=1).cpu().numpy()[0]
        pb = int(prob.argmax())
        conf = float(prob[pb])

        if pb >= n_bins:
            return 0.0, conf, n_bins

        lo = max(0, pb - 2)
        hi = min(n_bins - 1, pb + 2)
        wp = prob[lo:hi + 1]
        bi = np.arange(lo, hi + 1)
        hz = float(np.sum(wp * cqt_freqs[bi]) / (np.sum(wp) + 1e-9))
        return hz, conf, pb
