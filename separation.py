import pathlib
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf


root = pathlib.Path(__file__).parent / "separated"


def mix(paths, out):
    arrs = []
    sr = None

    for p in paths:
        p = pathlib.Path(p)

        if not p.exists():
            continue

        d, s = sf.read(str(p), always_2d=True)

        if sr is None:
            sr = s
        if s != sr:
            continue

        arrs.append(d)

    if not arrs:
        return None

    ml = max(a.shape[0] for a in arrs)
    ch = arrs[0].shape[1]
    mix = np.zeros((ml, ch), dtype=np.float32)

    for a in arrs:
        if a.shape[1] != ch:
            if a.shape[1] == 1 and ch == 2:
                a = np.repeat(a, 2, axis=1)
            elif a.shape[1] == 2 and ch == 1:
                a = a.mean(axis=1, keepdims=True)

        mix[:a.shape[0]] += a

    peak = np.max(np.abs(mix))

    if peak > 0.99:
        mix = mix / peak * 0.99

    sf.write(str(out), mix, sr)
    return out


def run_demucs(inp, outdir, model, two=None):
    py = sys.executable
    cmd = [py, "-m", "demucs.separate", "-n", model, "-o", str(outdir)]

    if two:
        cmd += ["--two-stems", two]

    cmd.append(str(inp))

    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(f"Demucs ({model}) failed: {r.stderr[-2000:]}")

    name = inp.stem
    d = outdir / model / name

    if not d.exists():
        cand = list(outdir.rglob("vocals.wav"))

        if cand:
            d = cand[0].parent
        else:
            cand = list(outdir.rglob("*.wav"))

            if cand:
                d = cand[0].parent
            else:
                raise FileNotFoundError(f"Demucs output not found under {outdir}")

    return d


def separate_song(inp, sid, coaching_type="vocals", model=None):
    inp = pathlib.Path(inp)

    if not inp.exists():
        raise FileNotFoundError(inp)

    coaching_type = (coaching_type or "vocals").lower()

    if coaching_type not in ("vocals", "guitar"):
        coaching_type = "vocals"

    outdir = pathlib.Path(__file__).parent / "separated"
    outdir.mkdir(parents=True, exist_ok=True)

    dest = pathlib.Path(__file__).parent / "separated" / str(sid)
    dest.mkdir(parents=True, exist_ok=True)

    mp = {}

    if coaching_type == "vocals":
        model = model or "htdemucs"
        d = run_demucs(inp, outdir, model, "vocals")
        voc = d / "vocals.wav"
        nov = d / "no_vocals.wav"

        if not nov.exists():
            mix([d / "drums.wav", d / "bass.wav", d / "other.wav"], dest / "instrumental.wav")

            if voc.exists():
                shutil.copy2(voc, dest / "vocals.wav")

            if (dest / "vocals.wav").exists():
                mp["vocals"] = dest / "vocals.wav"
            else:
                mp["vocals"] = None

            if (dest / "instrumental.wav").exists():
                mp["instrumental"] = dest / "instrumental.wav"
            else:
                mp["instrumental"] = None

            mp["backing"] = mp["instrumental"]
            mp["reference"] = mp["vocals"]
        else:
            if voc.exists():
                shutil.copy2(voc, dest / "vocals.wav")
                mp["vocals"] = dest / "vocals.wav"
            else:
                mp["vocals"] = None

            if nov.exists():
                shutil.copy2(nov, dest / "instrumental.wav")
                mp["instrumental"] = dest / "instrumental.wav"
            else:
                mp["instrumental"] = None

            mp["backing"] = mp["instrumental"]
            mp["reference"] = mp["vocals"]

        mp["guitar"] = None
        mp["other"] = None
    else:
        model = model or "htdemucs_6s"

        try:
            d = run_demucs(inp, outdir, model, None)
        except RuntimeError:
            if "htdemucs_6s" in str(model):
                d = run_demucs(inp, outdir, "htdemucs", None)
                model = "htdemucs"
            else:
                raise

        g = d / "guitar.wav"
        o = d / "other.wav"

        if g.exists():
            shutil.copy2(g, dest / "guitar.wav")
            mp["guitar"] = dest / "guitar.wav"
            mp["reference"] = dest / "guitar.wav"
        elif o.exists():
            shutil.copy2(o, dest / "other.wav")
            shutil.copy2(o, dest / "guitar.wav")
            mp["guitar"] = dest / "guitar.wav"
            mp["other"] = dest / "other.wav"
            mp["reference"] = dest / "guitar.wav"
        else:
            mp["guitar"] = None
            mp["other"] = None
            mp["reference"] = None

        v = d / "vocals.wav"

        if v.exists():
            shutil.copy2(v, dest / "vocals.wav")
            mp["vocals"] = dest / "vocals.wav"
        else:
            mp["vocals"] = None

        backs = []

        for cand in ["vocals", "drums", "bass", "other", "piano"]:
            p = d / f"{cand}.wav"

            if p.exists():
                try:
                    shutil.copy2(p, dest / f"{cand}.wav")
                except Exception:
                    pass
                backs.append(dest / f"{cand}.wav")

        if backs:
            mix(backs, dest / "instrumental.wav")
            mp["instrumental"] = dest / "instrumental.wav"
        else:
            nov = d / "no_vocals.wav"

            if nov.exists():
                shutil.copy2(nov, dest / "instrumental.wav")
                mp["instrumental"] = dest / "instrumental.wav"
            else:
                mp["instrumental"] = None

        mp["backing"] = mp["instrumental"]

    mp.setdefault("vocals", None)
    mp.setdefault("guitar", None)
    mp.setdefault("other", None)
    mp.setdefault("instrumental", None)
    mp.setdefault("backing", mp.get("instrumental"))
    mp.setdefault("reference", mp.get("vocals") or mp.get("guitar"))

    try:
        if d and pathlib.Path(d) != dest and pathlib.Path(d).exists():
            shutil.rmtree(str(d), ignore_errors=True)
    except Exception:
        pass

    return mp


def get_stem_paths(sid):
    base = pathlib.Path(__file__).parent / "separated" / str(sid)

    def p(n):
        q = base / f"{n}.wav"
        if q.exists():
            return q
        return None

    return {
        "vocals": p("vocals"),
        "guitar": p("guitar"),
        "other": p("other"),
        "instrumental": p("instrumental"),
        "backing": p("instrumental"),
        "reference": p("guitar") or p("vocals") or p("other"),
        "drums": p("drums"),
        "bass": p("bass"),
    }
