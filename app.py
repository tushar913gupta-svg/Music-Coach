import datetime
import json
import os
import pathlib
import secrets
import shutil
import threading
import uuid

import librosa
import numpy as np
import soundfile as sf

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import inference
import lyrics
from separation import get_stem_paths, separate_song


basedir = pathlib.Path(__file__).parent.resolve()
uploaddir = basedir / "uploads"
separatedir = basedir / "separated"
recorddir = basedir / "recordings"
instancedir = basedir / "instance"

for d in [uploaddir, separatedir, recorddir, instancedir]:
    d.mkdir(parents=True, exist_ok=True)


def user_record_dir(user_id):
    d = recorddir / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

allowed = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{instancedir / 'music_coach.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(uploaddir)

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"


class User(UserMixin, db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    songs = db.relationship("Song", back_populates="owner", cascade="all, delete-orphan", lazy=True)
    attempts = db.relationship("Attempt", back_populates="author", cascade="all, delete-orphan", lazy=True)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class Song(db.Model):
    __tablename__ = "song"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(30), default="pending", nullable=False)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    duration = db.Column(db.Float, nullable=True)
    coaching_type = db.Column(db.String(20), default="vocals", nullable=False)
    target_score = db.Column(db.Integer, nullable=True)

    owner = db.relationship("User", back_populates="songs")
    attempts = db.relationship(
        "Attempt",
        back_populates="song",
        cascade="all, delete-orphan",
        lazy=True,
        order_by="Attempt.created_at.desc()",
    )

    @property
    def stem_paths(self):
        return get_stem_paths(self.id)

    @property
    def has_vocals(self):
        return (separatedir / str(self.id) / "vocals.wav").exists()

    @property
    def has_guitar(self):
        p = separatedir / str(self.id) / "guitar.wav"
        return p.exists() or (separatedir / str(self.id) / "other.wav").exists()

    @property
    def has_instrumental(self):
        return (separatedir / str(self.id) / "instrumental.wav").exists()

    @property
    def reference_label(self):
        if self.coaching_type == "guitar":
            return "Guitar"
        return "Vocals"

    def reference_path(self):
        base = separatedir / str(self.id)

        if self.coaching_type == "guitar":
            candidates = ["guitar.wav", "other.wav", "vocals.wav"]
        else:
            candidates = ["vocals.wav", "other.wav", "guitar.wav"]

        for name in candidates:
            p = base / name
            if p.exists():
                return p

        return pathlib.Path(self.filepath)


class Attempt(db.Model):
    __tablename__ = "attempt"

    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.Integer, db.ForeignKey("song.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    recording_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    pitch_score = db.Column(db.Integer, nullable=True)
    octave_score = db.Column(db.Integer, nullable=True)
    timing_score = db.Column(db.Integer, nullable=True)
    overall = db.Column(db.Integer, nullable=True)
    timing_reliable = db.Column(db.Boolean, nullable=True)
    median_cents = db.Column(db.Float, nullable=True)
    median_octave = db.Column(db.Float, nullable=True)
    mean_cents = db.Column(db.Float, nullable=True)
    median_onset = db.Column(db.Float, nullable=True)
    mean_onset = db.Column(db.Float, nullable=True)
    tempo_ref = db.Column(db.Float, nullable=True)
    tempo_rec = db.Column(db.Float, nullable=True)
    voiced_frames = db.Column(db.Integer, nullable=True)
    onsets_ref = db.Column(db.Integer, nullable=True)
    onsets_rec = db.Column(db.Integer, nullable=True)
    issues_json = db.Column(db.Text, nullable=True)

    song = db.relationship("Song", back_populates="attempts")
    author = db.relationship("User", back_populates="attempts")

    @property
    def issues(self):
        try:
            if self.issues_json:
                return json.loads(self.issues_json)
            return []
        except Exception:
            return []


album_songs = db.Table(
    "album_song",
    db.Column("album_id", db.Integer, db.ForeignKey("album.id", ondelete="CASCADE"), primary_key=True),
    db.Column("song_id", db.Integer, db.ForeignKey("song.id", ondelete="CASCADE"), primary_key=True),
)


class Album(db.Model):
    __tablename__ = "album"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    songs = db.relationship(
        "Song",
        secondary=album_songs,
        backref="albums",
        order_by="Song.created_at.desc()",
    )


@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))


@app.context_processor
def inject_albums():
    if current_user.is_authenticated:
        try:
            albums = Album.query.filter_by(user_id=current_user.id).order_by(Album.created_at.desc()).all()
        except Exception:
            albums = []
        return {"user_albums": albums}
    return {"user_albums": []}


def allowed_file(fname):
    return pathlib.Path(fname).suffix.lower() in allowed


def background_separation(sid, path):
    with app.app_context():
        song = Song.query.get(sid)

        if not song:
            return

        song.status = "separating"
        db.session.commit()

        try:
            coaching = song.coaching_type or "vocals"
            mapping = separate_song(path, sid, coaching_type=coaching)

            stems = ["vocals", "guitar", "other", "instrumental"]
            found = False
            for key in stems:
                p = mapping.get(key)
                if p and pathlib.Path(p).exists():
                    found = True
                    break

            if not found:
                raise RuntimeError("No stems produced")

            song.status = "done"
            song.error = None

            if coaching == "vocals":
                try:
                    vpath = mapping.get("vocals")
                    if vpath and pathlib.Path(vpath).exists():
                        lyrics.transcribe_song(sid, vpath)
                except Exception:
                    import traceback

                    traceback.print_exc()
        except Exception as e:
            import traceback

            traceback.print_exc()
            song.status = "failed"
            song.error = str(e)[:2000]

        db.session.commit()


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip() or None
        pw = request.form.get("password", "")
        pw2 = request.form.get("password2", "")

        if not username or not pw:
            flash("Please complete all fields.", "danger")
        elif pw != pw2:
            flash("Passwords do not match.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("Username already taken.", "warning")
        elif email and User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "warning")
        else:
            try:
                u = User(username=username, email=email, password_hash="")
                u.set_password(pw)
                db.session.add(u)
                db.session.commit()
                login_user(u)
                flash("Account created.", "success")
                return redirect(url_for("index"))
            except IntegrityError:
                db.session.rollback()
                flash("Username or email already taken.", "warning")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        pw = request.form.get("password", "")

        if not username or not pw:
            flash("Please complete all fields.", "danger")
            return redirect(url_for("login"))

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("No account found with that username.", "warning")
            return redirect(url_for("login"))

        if not user.check_password(pw):
            flash("Incorrect password.", "danger")
            return redirect(url_for("login"))

        login_user(user, remember=True)
        nxt = request.args.get("next")
        return redirect(nxt or url_for("index"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    songs = Song.query.filter_by(user_id=current_user.id).order_by(Song.created_at.desc()).all()

    week_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    takes_week = Attempt.query.filter_by(user_id=current_user.id).filter(Attempt.created_at >= week_ago).count()

    best = db.session.query(db.func.max(Attempt.overall)).filter_by(user_id=current_user.id).scalar()

    days = db.session.query(db.func.date(Attempt.created_at)).filter_by(user_id=current_user.id).distinct().all()
    streak = len(days)

    recent = {}
    for s in songs:
        last = Attempt.query.filter_by(song_id=s.id).order_by(Attempt.created_at.desc()).limit(3).all()
        recent[s.id] = [a.overall for a in reversed(last) if a.overall is not None]

    return render_template(
        "index.html",
        songs=songs,
        total_songs=len(songs),
        takes_week=takes_week,
        best_total=best or 0,
        streak=streak,
        recent=recent,
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected.", "danger")
            return redirect(request.url)

        f = request.files["file"]

        if f.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not allowed_file(f.filename):
            flash("Unsupported file type.", "danger")
            return redirect(request.url)

        title = request.form.get("title", "").strip() or pathlib.Path(f.filename).stem

        ctype = request.form.get("coaching_type", "vocals").strip().lower()
        if ctype not in ("vocals", "guitar"):
            ctype = "vocals"

        ext = pathlib.Path(f.filename).suffix.lower()
        safe = secure_filename(pathlib.Path(f.filename).stem)[:50]
        uniq = f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
        dest = uploaddir / uniq
        f.save(str(dest))

        duration = None
        try:
            duration = sf.info(str(dest)).duration
        except Exception:
            pass

        song = Song(
            user_id=current_user.id,
            title=title,
            filename=uniq,
            filepath=str(dest),
            status="pending",
            duration=duration,
            coaching_type=ctype,
        )
        db.session.add(song)
        db.session.commit()

        threading.Thread(target=background_separation, args=(song.id, dest), daemon=True).start()

        flash("Upload complete. Processing audio.", "success")
        return redirect(url_for("song_detail", song_id=song.id))

    return render_template("upload.html")


@app.route("/song/<int:song_id>")
@login_required
def song_detail(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    attempts = Attempt.query.filter_by(song_id=song.id).order_by(Attempt.created_at.desc()).all()
    ordered = list(reversed(attempts))

    progress = [
        {"date": a.created_at.strftime("%m-%d"), "overall": a.overall}
        for a in ordered
        if a.overall is not None
    ]

    best = None
    for a in attempts:
        if a.overall is not None and (best is None or a.overall > best):
            best = a.overall

    return render_template("song.html", song=song, attempts=attempts, progress=progress, best=best)


@app.route("/song/<int:song_id>/target", methods=["POST"])
@login_required
def set_target(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    raw = request.form.get("target_score", "").strip()

    if raw == "":
        song.target_score = None
    else:
        try:
            val = int(raw)
            song.target_score = max(0, min(100, val))
        except ValueError:
            flash("Target must be a number 0-100.", "danger")
            return redirect(url_for("song_detail", song_id=song.id))

    db.session.commit()
    flash("Target saved.", "success")
    return redirect(url_for("song_detail", song_id=song.id))


@app.route("/practice/<int:song_id>")
@login_required
def practice(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    if song.status != "done":
        flash("Audio is still processing.", "warning")
        return redirect(url_for("song_detail", song_id=song.id))

    return render_template("practice.html", song=song)


@app.route("/results/<int:attempt_id>")
@login_required
def results(attempt_id):
    att = Attempt.query.get(attempt_id)

    if not att or att.user_id != current_user.id:
        abort(404)

    song = Song.query.get(att.song_id)

    song_takes = Attempt.query.filter_by(song_id=song.id).order_by(Attempt.created_at.desc()).all()

    best = None
    for a in song_takes:
        if a.overall is not None and (best is None or a.overall > best):
            best = a.overall

    is_best = best is not None and att.overall is not None and att.overall >= best and len(song_takes) > 1

    history = [a for a in song_takes if a.id != att.id][:5]

    return render_template("results.html", attempt=att, song=song, best=best, is_best=is_best, history=history)


@app.route("/delete_take/<int:attempt_id>", methods=["POST"])
@login_required
def delete_take(attempt_id):
    att = Attempt.query.get(attempt_id)

    if not att or att.user_id != current_user.id:
        abort(404)

    song_id = att.song_id

    try:
        if att.recording_path and pathlib.Path(att.recording_path).exists():
            pathlib.Path(att.recording_path).unlink(missing_ok=True)
    except Exception:
        pass

    db.session.delete(att)
    db.session.commit()

    flash("Take deleted.", "info")
    return redirect(url_for("song_detail", song_id=song_id))


@app.route("/delete/<int:song_id>", methods=["POST"])
@login_required
def delete_song(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    try:
        if song.filepath and pathlib.Path(song.filepath).exists():
            pathlib.Path(song.filepath).unlink(missing_ok=True)
    except Exception:
        pass

    sdir = separatedir / str(song.id)
    if sdir.exists():
        shutil.rmtree(str(sdir), ignore_errors=True)

    for att in song.attempts:
        if att.recording_path and pathlib.Path(att.recording_path).exists():
            try:
                pathlib.Path(att.recording_path).unlink(missing_ok=True)
            except Exception:
                pass

    try:
        db.session.execute(album_songs.delete().where(album_songs.c.song_id == song.id))
    except Exception:
        pass

    db.session.delete(song)
    db.session.commit()

    flash("Song deleted.", "info")
    return redirect(url_for("index"))


@app.route("/albums", methods=["POST"])
@login_required
def create_album():
    name = request.form.get("name", "").strip()[:120]

    if not name:
        flash("Name your album.", "warning")
        return redirect(request.referrer or url_for("index"))

    alb = Album(user_id=current_user.id, name=name)
    db.session.add(alb)
    db.session.commit()

    flash("Album created.", "success")
    return redirect(url_for("album_detail", album_id=alb.id))


@app.route("/album/<int:album_id>")
@login_required
def album_detail(album_id):
    alb = Album.query.get(album_id)

    if not alb or alb.user_id != current_user.id:
        abort(404)

    all_songs = Song.query.filter_by(user_id=current_user.id).order_by(Song.created_at.desc()).all()

    return render_template("album.html", album=alb, songs=alb.songs, all_songs=all_songs)


@app.route("/albums/add", methods=["POST"])
@login_required
def albums_add():
    try:
        album_id = int(request.form.get("album_id", 0))
        song_id = int(request.form.get("song_id", 0))
    except (TypeError, ValueError):
        abort(404)

    alb = Album.query.get(album_id)
    song = Song.query.get(song_id)

    if not alb or alb.user_id != current_user.id:
        abort(404)
    if not song or song.user_id != current_user.id:
        abort(404)

    if song not in alb.songs:
        alb.songs.append(song)
        db.session.commit()
        flash(f"Added to {alb.name}.", "success")

    return redirect(request.referrer or url_for("album_detail", album_id=alb.id))


@app.route("/album/<int:album_id>/remove", methods=["POST"])
@login_required
def album_remove(album_id):
    alb = Album.query.get(album_id)

    if not alb or alb.user_id != current_user.id:
        abort(404)

    try:
        song_id = int(request.form.get("song_id", 0))
    except (TypeError, ValueError):
        song_id = 0

    song = Song.query.get(song_id)

    if song and song in alb.songs:
        alb.songs.remove(song)
        db.session.commit()
        flash(f"Removed from {alb.name}.", "info")

    return redirect(request.referrer or url_for("album_detail", album_id=alb.id))


@app.route("/album/<int:album_id>/delete", methods=["POST"])
@login_required
def delete_album(album_id):
    alb = Album.query.get(album_id)

    if not alb or alb.user_id != current_user.id:
        abort(404)

    try:
        db.session.execute(album_songs.delete().where(album_songs.c.album_id == alb.id))
    except Exception:
        pass

    db.session.delete(alb)
    db.session.commit()

    flash("Album deleted.", "info")
    return redirect(url_for("index"))


@app.route("/api/status/<int:song_id>")
@login_required
def api_status(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        return jsonify({"error": "not found"}), 404

    return jsonify({
        "status": song.status,
        "error": song.error,
        "coaching_type": song.coaching_type,
        "reference_label": song.reference_label,
        "has_vocals": song.has_vocals,
        "has_guitar": song.has_guitar,
        "has_instrumental": song.has_instrumental,
    })


@app.route("/api/lyrics/<int:song_id>")
@login_required
def api_lyrics(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        return jsonify({"error": "not found"}), 404

    if (song.coaching_type or "vocals") != "vocals":
        return jsonify({"segments": [], "language": None})

    p = lyrics.lyrics_path(song_id)

    try:
        if p.exists():
            return jsonify(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass

    return jsonify({"segments": [], "language": None, "pending": song.status != "done"})


@app.route("/api/reference_pitch/<int:song_id>")
@login_required
def api_reference_pitch(song_id):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        return jsonify({"error": "not found"}), 404

    ref = song.reference_path()
    if not ref.exists():
        ref = pathlib.Path(song.filepath)

    if not ref.exists():
        return jsonify({"error": "reference not found"}), 404

    try:
        times, hz, conf, _ = inference.predict_frames(inference.load_audio(str(ref)))
        s = 2
        return jsonify({
            "times": times[::s].tolist(),
            "hz": hz[::s].tolist(),
            "conf": conf[::s].tolist(),
            "sr": inference.sr,
            "hop_time": inference.hop_time * s,
        })
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pitch", methods=["POST"])
@login_required
def api_pitch():
    y = None
    sr_in = inference.sr

    if request.is_json:
        data = request.get_json()
        samples = data.get("samples") or data.get("pcm") or []

        if not samples:
            return jsonify({"error": "no samples"}), 400

        y = np.array(samples, dtype=np.float32)
        sr_claim = int(data.get("sampleRate", 16000) or data.get("sr", 16000))

        if sr_claim != inference.sr and len(y) > 0:
            y = librosa.resample(y, orig_sr=sr_claim, target_sr=inference.sr)
    else:
        if "audio" not in request.files and "file" not in request.files:
            return jsonify({"error": "no audio file"}), 400

        f = request.files.get("audio") or request.files.get("file")
        tmp = recorddir / f"_tmp_{uuid.uuid4().hex}.wav"
        f.save(str(tmp))

        try:
            y = inference.load_audio(str(tmp), sr_in)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    if y is None or len(y) == 0:
        return jsonify({"error": "empty audio"}), 400

    try:
        times, hz, conf, _ = inference.predict_frames(y)
        th = float(request.args.get("th", inference.conf_th_calib))
        hzf = [float(h) if float(c) >= th else 0 for h, c in zip(hz, conf)]
        return jsonify({
            "times": times.tolist(),
            "hz": hzf,
            "conf": conf.tolist(),
            "voiced": (conf >= th).tolist(),
        })
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_pcm", methods=["POST"])
@login_required
def api_save_pcm():
    if not request.is_json:
        return jsonify({"error": "expected JSON"}), 400

    data = request.get_json()
    samples = data.get("samples") or data.get("pcm")

    if samples is None:
        return jsonify({"error": "no samples"}), 400

    sr_claim = int(data.get("sampleRate", 16000) or 16000)
    y = np.array(samples, dtype=np.float32)

    if sr_claim != inference.sr:
        y = librosa.resample(y, orig_sr=sr_claim, target_sr=inference.sr)
        sr_claim = inference.sr

    fname = f"pcm_{current_user.id}_{uuid.uuid4().hex}.wav"
    out = user_record_dir(current_user.id) / fname
    sf.write(str(out), y, sr_claim)

    return jsonify({"saved": str(out), "samples": len(y), "sr": sr_claim})


@app.route("/api/score", methods=["POST"])
@login_required
def api_score():
    sid = None

    if request.is_json:
        sid = (request.get_json(silent=True) or {}).get("song_id")
    if sid is None:
        sid = request.form.get("song_id") or request.args.get("song_id")

    if sid is None:
        return jsonify({"error": "song_id required"}), 400

    try:
        sid = int(sid)
    except Exception:
        return jsonify({"error": "invalid song_id"}), 400

    song = Song.query.get(sid)

    if not song or song.user_id != current_user.id:
        return jsonify({"error": "song not found"}), 404

    ref = song.reference_path()
    if not ref.exists():
        ref = pathlib.Path(song.filepath)

    if not ref.exists():
        return jsonify({"error": "reference audio missing"}), 500

    tmp_rec = user_record_dir(current_user.id) / f"rec_{current_user.id}_{sid}_{uuid.uuid4().hex}.wav"

    if request.is_json:
        data = request.get_json()
        samples = data.get("samples") or data.get("pcm") or data.get("audio")

        if samples is not None and isinstance(samples, list):
            y = np.array(samples, dtype=np.float32)
            sr_claim = int(data.get("sampleRate", 16000) or 16000)

            if sr_claim != inference.sr:
                y = librosa.resample(y, orig_sr=sr_claim, target_sr=inference.sr)
                sr_claim = inference.sr

            sf.write(str(tmp_rec), y, sr_claim)
        else:
            return jsonify({"error": "no audio samples in JSON"}), 400
    else:
        f = request.files.get("audio") or request.files.get("file") or request.files.get("recording")

        if f:
            tmp_raw = recorddir / f"_raw_{uuid.uuid4().hex}{pathlib.Path(f.filename).suffix or '.wav'}"
            f.save(str(tmp_raw))

            try:
                y = inference.load_audio(str(tmp_raw), inference.sr)
                sf.write(str(tmp_rec), y, inference.sr)
                tmp_raw.unlink(missing_ok=True)
            except Exception as e:
                import traceback

                traceback.print_exc()
                return jsonify({"error": f"failed to process audio: {e}"}), 500
        else:
            if request.data and len(request.data) > 100:
                tmp_raw = recorddir / f"_body_{uuid.uuid4().hex}.wav"
                tmp_raw.write_bytes(request.data)

                try:
                    y = inference.load_audio(str(tmp_raw), inference.sr)
                    sf.write(str(tmp_rec), y, inference.sr)
                    tmp_raw.unlink(missing_ok=True)
                except Exception as e:
                    return jsonify({"error": f"invalid audio body: {e}"}), 400
            else:
                return jsonify({"error": "no audio provided"}), 400

    try:
        report, ref_data, rec_data = inference.practice_report(str(ref), str(tmp_rec))

        att = Attempt(
            song_id=song.id,
            user_id=current_user.id,
            recording_path=str(tmp_rec),
            pitch_score=report["Pitch Score"],
            octave_score=report["Octave Score"],
            timing_score=report["Timing Score"],
            overall=report["Overall"],
            timing_reliable=report["Timing Reliable"],
            median_cents=report["Median Cents"],
            median_octave=report["Median Octave"],
            mean_cents=report["Mean Cents"],
            median_onset=report["Median Onset (ms)"],
            mean_onset=report["Mean Onset (ms)"],
            tempo_ref=report["Tempo Ref"],
            tempo_rec=report["Tempo Rec"],
            voiced_frames=report["Voiced Frames"],
            onsets_ref=report["Onsets Ref"],
            onsets_rec=report["Onsets Rec"],
            issues_json=json.dumps(report["Issues"]),
        )
        db.session.add(att)
        db.session.commit()

        tr, hr, cr = ref_data
        tp, hp, cp = rec_data
        s = 2

        return jsonify({
            "attempt_id": att.id,
            "report": report,
            "ref": {
                "times": tr[::s].tolist(),
                "hz": hr[::s].tolist(),
                "conf": cr[::s].tolist(),
            },
            "rec": {
                "times": tp[::s].tolist(),
                "hz": hp[::s].tolist(),
                "conf": cp[::s].tolist(),
            },
        })
    except Exception as e:
        import traceback

        traceback.print_exc()

        try:
            tmp_rec.unlink(missing_ok=True)
        except Exception:
            pass

        return jsonify({"error": str(e)}), 500


@app.route("/api/recording/<int:attempt_id>")
@login_required
def api_recording_pitch(attempt_id):
    att = Attempt.query.get(attempt_id)

    if not att or att.user_id != current_user.id:
        return jsonify({"error": "not found"}), 404

    rec = pathlib.Path(att.recording_path) if att.recording_path else None

    if not rec or not rec.exists():
        return jsonify({"error": "recording missing"}), 404

    try:
        times, hz, conf, _ = inference.predict_frames(inference.load_audio(str(rec)))
        s = 2
        return jsonify({
            "times": times[::s].tolist(),
            "hz": hz[::s].tolist(),
            "conf": conf[::s].tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/<kind>/<int:song_id>/<fname>")
@login_required
def serve_audio(kind, song_id, fname):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    if kind == "upload":
        return send_from_directory(str(uploaddir), fname)
    elif kind == "separated":
        base = separatedir / str(song_id)
        return send_from_directory(str(base), fname)
    else:
        abort(404)


@app.route("/audio/recording/<int:attempt_id>")
@login_required
def serve_recording(attempt_id):
    att = Attempt.query.get(attempt_id)

    if not att or att.user_id != current_user.id:
        abort(404)

    p = pathlib.Path(att.recording_path)

    if not p.exists():
        abort(404)

    return send_from_directory(str(p.parent), p.name)


@app.route("/audio/separated/<int:song_id>/<stem>")
@login_required
def serve_stem(song_id, stem):
    song = Song.query.get(song_id)

    if not song or song.user_id != current_user.id:
        abort(404)

    base = separatedir / str(song_id)
    fname = f"{stem}.wav"

    if not (base / fname).exists():
        abort(404)

    return send_from_directory(str(base), fname)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "model": str(inference.mpath.exists()),
        "device": str(inference.device),
    })


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(port=5000, debug=True, threaded=True)
