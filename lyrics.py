import json
import pathlib

from faster_whisper import WhisperModel

model_size = "base"


def lyrics_path(sid):
    return pathlib.Path(__file__).parent / "separated" / str(sid) / "lyrics.json"


def transcribe_file(vocals_path):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(vocals_path), vad_filter=True)

    out = []

    for sg in segments:
        words = []

        for w in (sg.words or []):
            text = (w.word or "").strip()
            if text:
                words.append({
                    "w": text,
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                })

        if not words and (sg.text or "").strip():
            words = [{
                "w": sg.text.strip(),
                "start": round(float(sg.start), 2),
                "end": round(float(sg.end), 2),
            }]

        if not words:
            continue

        out.append({
            "start": round(float(sg.start), 2),
            "end": round(float(sg.end), 2),
            "text": (sg.text or "").strip(),
            "words": words,
        })

    return {"language": getattr(info, "language", None), "segments": out}


def transcribe_song(sid, vocals_path):
    data = transcribe_file(vocals_path)
    dest = lyrics_path(sid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data), encoding="utf-8")
    return data
