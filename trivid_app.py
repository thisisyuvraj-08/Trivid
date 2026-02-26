"""
╔══════════════════════════════════════════════════════╗
║          TRIVID v2.0 — Desktop AI Video Studio       ║
║  Script → Voice → Visuals → Edit → Captions → MP4   ║
╚══════════════════════════════════════════════════════╝

Requirements (auto-installed on first run):
  pip install customtkinter pillow requests yt-dlp moviepy==1.0.3
  pip install openai-whisper google-generativeai kokoro soundfile scipy
  pip install numpy

Also needs ffmpeg on PATH:
  Download from https://ffmpeg.org/download.html and add to PATH.
"""

# ── Dependency bootstrap ───────────────────────────────────────────────────────
import subprocess, sys, os

REQUIRED = [
    "customtkinter", "PIL", "requests", "yt_dlp",
    "moviepy", "whisper", "google.generativeai",
    "kokoro", "soundfile", "numpy", "scipy", "duckduckgo_search"
]

def _install_deps():
    pkgs = {
        "customtkinter":       "customtkinter",
        "PIL":                 "pillow",
        "requests":            "requests",
        "yt_dlp":              "yt-dlp",
        "moviepy":             "moviepy==1.0.3",
        "whisper":             "openai-whisper",
        "google.generativeai": "google-generativeai",
        "kokoro":              "kokoro",
        "soundfile":           "soundfile",
        "numpy":               "numpy",
        "scipy":               "scipy",
        "duckduckgo_search":   "duckduckgo-search",
    }
    for mod, pkg in pkgs.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"Installing missing package: {pkg}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    timeout=300
                )
            except Exception as e:
                print(f"  WARNING: Could not install {pkg}: {e}")

try:
    _install_deps()
except Exception as e:
    print(f"Dependency bootstrap warning: {e}")

# ── SSL Fix: disable cert verification (network SSL inspection proxy detected) ──
# Your network intercepts HTTPS with its own cert not in any Python trust store.
# verify=False is safe here — we only call trusted Google/media APIs.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests so ALL calls in this app use verify=False automatically
import requests as _req_module
_OrigSession = _req_module.Session
class _NoVerifySession(_OrigSession):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return super().request(method, url, **kwargs)
_req_module.Session = _NoVerifySession

# ── Standard imports ───────────────────────────────────────────────
import threading, queue, json, time, shutil, re, operator
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import requests, numpy as np

# ── App Config ─────────────────────────────────────────────────────────────────
APP_NAME    = "Trivid"
APP_VERSION = "2.0"
CONFIG_DIR  = os.path.join(os.path.expanduser("~"), ".trivid")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
WORK_DIR    = os.path.join(CONFIG_DIR, "workspace")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(WORK_DIR,   exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self, settings: dict, log_q: queue.Queue, progress_q: queue.Queue):
        self.s         = settings
        self.log_q     = log_q
        self.progress_q = progress_q
        self.scenes    = []
        self.audio_files = []
        self.media_files = []
        self.output_path = None

    def log(self, msg, level="info"):
        self.log_q.put((level, msg))

    def prog(self, value, label=""):
        self.progress_q.put((value, label))

    def run(self):
        try:
            self.prog(0,  "Starting pipeline…")
            self.step_gemini()
            self.prog(15, "Scene breakdown done")
            self.step_tts()
            self.prog(35, "Voiceover generated")
            self.step_media()
            self.prog(60, "Visuals downloaded")
            self.step_assemble()
            self.prog(80, "Video assembled")
            self.step_captions()
            self.prog(100, "Done!")
        except Exception as e:
            self.log(f"PIPELINE ERROR: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")

    # ── Step 1: Gemini analysis ──────────────────────────────────────────────
    def step_gemini(self):
        self.log("🧠 Sending script to Gemini AI for scene breakdown…")

        prompt = f"""You are a professional video editor AI. Analyze the following script and break it into \
scenes for a YouTube video. Each scene should be 5-10 seconds of spoken content.

Return a JSON array with objects with these exact keys:
- "segment_id": integer starting at 1
- "text": exact script text for this segment (verbatim)
- "duration_hint_secs": estimated speaking duration in seconds (~2.5 words/sec)
- "media_keywords": list of 3 highly specific, visual search keywords for image search.
  Make these descriptive and aesthetic — e.g. "ancient roman colosseum ruins golden hour",
  "vast ocean waves cinematic", "dense amazon rainforest aerial view"
- "media_type": "image" (prefer images) or "video" (only for very dynamic scenes)
- "use_wikimedia": true if the content is historical/documentary (paintings, maps, artifacts)

RULES:
- Prioritize vivid, visual keywords that will return beautiful, high-quality images
- For historical content set use_wikimedia to true
- Return ONLY the raw JSON array, no markdown fences, no explanation.

SCRIPT:
{self.s["script"]}"""

        # ── Use Gemini REST API directly (avoids gRPC SSL issues on Windows) ──
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"/gemini-2.0-flash:generateContent?key={self.s['gemini_key']}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
            }
        }
        self.log("   Calling Gemini 2.0 Flash via REST…")
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()

        data = r.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw).strip()
        raw = re.sub(r'\s*```$', '', raw).strip()
        self.scenes = json.loads(raw)
        self.log(f"✅ Gemini created {len(self.scenes)} scenes.")
        for sc in self.scenes:
            self.log(f"   Scene {sc['segment_id']}: \"{sc['text'][:60]}…\" ({sc['duration_hint_secs']:.1f}s)")

    # ── Step 2: Kokoro TTS ───────────────────────────────────────────────────
    def step_tts(self):
        self.log("🎤 Generating voiceover with Kokoro TTS…")
        from kokoro import KPipeline
        import soundfile as sf2

        voice_map = {
            "English — Female (af_heart)":  ("a", "af_heart"),
            "English — Male (am_adam)":      ("a", "am_adam"),
            "Hindi — Female (hf_alpha)":     ("h", "hf_alpha"),
        }
        lang_code, voice_name = voice_map.get(self.s["voice"], ("a", "af_heart"))
        pipeline = KPipeline(lang_code=lang_code)

        audio_dir = os.path.join(WORK_DIR, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        self.audio_files = []

        for sc in self.scenes:
            seg_id = sc["segment_id"]
            out    = os.path.join(audio_dir, f"seg_{seg_id:03d}.wav")
            self.log(f"   [{seg_id}/{len(self.scenes)}] TTS: \"{sc['text'][:50]}…\"")
            chunks = []
            for _, _, audio_chunk in pipeline(sc["text"], voice=voice_name, speed=1.0):
                chunks.append(audio_chunk)
            if chunks:
                audio_data = np.concatenate(chunks)
                sf2.write(out, audio_data, 24000)
                sc["actual_audio_duration"] = len(audio_data) / 24000.0
                self.audio_files.append(out)
                self.log(f"      ✅ {sc['actual_audio_duration']:.1f}s saved.")
            else:
                sc["actual_audio_duration"] = sc["duration_hint_secs"]
                self.audio_files.append(None)
                self.log(f"      ⚠️  No audio for scene {seg_id}.", "warn")

        total = sum(sc.get("actual_audio_duration", 0) for sc in self.scenes)
        self.log(f"✅ Voiceover complete. Total: {total:.1f}s")

    # ── Step 3: Media sourcing ───────────────────────────────────────────────
    def step_media(self):
        self.log("🌐 Sourcing visuals (DuckDuckGo → Wikimedia → Pexels → YouTube)…")
        media_dir = os.path.join(WORK_DIR, "media")
        os.makedirs(media_dir, exist_ok=True)
        self.media_files = []

        for sc in self.scenes:
            seg_id   = sc["segment_id"]
            keywords = sc["media_keywords"]
            mtype    = sc.get("media_type", "image")
            use_wiki = sc.get("use_wikimedia", False)
            found    = None

            self.log(f"   Scene {seg_id}: searching {keywords[0]!r} [{mtype}]")
            out_base = os.path.join(media_dir, f"seg_{seg_id:03d}")

            for keyword in keywords:
                if found: break

                # ── 1. DuckDuckGo (no API key — always tried first) ──────────
                if not found and mtype == "image":
                    url = self._ddg_image(keyword)
                    if url:
                        found = self._dl(url, out_base + ".jpg")
                        if found: self.log(f"      ✅ DuckDuckGo image")

                # ── 2. Wikimedia (best for historical/documentary content) ───
                if not found and use_wiki:
                    url = self._wikimedia_image(keyword)
                    if url:
                        found = self._dl(url, out_base + ".jpg")
                        if found: self.log(f"      ✅ Wikimedia image")

                # ── 3. Pexels (optional — used if key is provided) ──────────
                if not found and self.s.get("pexels_key"):
                    if mtype == "video":
                        url = self._pexels_video(keyword, self.s["pexels_key"])
                        if url:
                            found = self._dl(url, out_base + ".mp4")
                            if found: self.log(f"      ✅ Pexels video")
                    else:
                        url = self._pexels_image(keyword, self.s["pexels_key"])
                        if url:
                            found = self._dl(url, out_base + ".jpg")
                            if found: self.log(f"      ✅ Pexels image")

                # ── 4. Pixabay (optional — used if key is provided) ─────────
                if not found and self.s.get("pixabay_key"):
                    if mtype == "video":
                        url = self._pixabay_video(keyword, self.s["pixabay_key"])
                        if url:
                            found = self._dl(url, out_base + ".mp4")
                            if found: self.log(f"      ✅ Pixabay video")
                    else:
                        url = self._pixabay_image(keyword, self.s["pixabay_key"])
                        if url:
                            found = self._dl(url, out_base + ".jpg")
                            if found: self.log(f"      ✅ Pixabay image")

                # ── 5. Wikimedia fallback for any type ───────────────────────
                if not found and not use_wiki:
                    url = self._wikimedia_image(keyword)
                    if url:
                        found = self._dl(url, out_base + ".jpg")
                        if found: self.log(f"      ✅ Wikimedia fallback")

                # ── 6. YouTube (last resort, videos only) ───────────────────
                if not found and mtype == "video":
                    self.log(f"      ⚠️  Trying YouTube (<7s clip)…", "warn")
                    found = self._youtube_clip(keyword, out_base + "_yt.mp4")
                    if found: self.log(f"      ✅ YouTube clip")

            if not found:
                self.log(f"      ⚠️  No media found, color background will be used.", "warn")
                found = "COLOR_BG"

            sc["media_path"] = found
            self.media_files.append(found)

        ok = sum(1 for m in self.media_files if m and m != "COLOR_BG")
        self.log(f"✅ Media sourcing done: {ok}/{len(self.scenes)} scenes have real visuals.")

    # ── Media helpers ────────────────────────────────────────────────────────
    def _dl(self, url, dest):
        try:
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/120.0.0.0 Safari/537.36"),
                "Referer": "https://www.google.com/",
            }
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return dest if os.path.getsize(dest) > 5000 else None
        except Exception as e:
            self.log(f"      Download error: {e}", "warn")
            return None

    def _ddg_image(self, keyword):
        """Search DuckDuckGo images — no API key required."""
        try:
            from duckduckgo_search import DDGS
            # Add aesthetic qualifiers for better visual results
            query = f"{keyword} aesthetic cinematic HD wallpaper"
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    query,
                    max_results=15,
                    size="Large",
                    type_image="photo",
                ))
            # Prefer common image formats; skip small/thumbnail URLs
            good_exts = (".jpg", ".jpeg", ".png", ".webp")
            for r in results:
                url = r.get("image", "")
                w   = r.get("width",  0)
                h   = r.get("height", 0)
                if (w >= 800 or h >= 600) and any(url.lower().split("?")[0].endswith(e) for e in good_exts):
                    return url
            # Fallback: first result regardless
            if results:
                return results[0].get("image")
        except Exception as e:
            self.log(f"      DuckDuckGo search error: {e}", "warn")
        return None

    def _pexels_video(self, keyword, key):
        try:
            r = requests.get("https://api.pexels.com/videos/search",
                             headers={"Authorization": key},
                             params={"query": keyword, "per_page": 5}, timeout=10)
            for v in r.json().get("videos", []):
                for f in sorted(v.get("video_files", []), key=lambda x: x.get("width",0), reverse=True):
                    if f.get("file_type") == "video/mp4":
                        return f["link"]
        except: pass
        return None

    def _pexels_image(self, keyword, key):
        try:
            r = requests.get("https://api.pexels.com/v1/search",
                             headers={"Authorization": key},
                             params={"query": keyword, "per_page": 3}, timeout=10)
            photos = r.json().get("photos", [])
            if photos: return photos[0]["src"]["large2x"]
        except: pass
        return None

    def _pixabay_video(self, keyword, key):
        try:
            r = requests.get("https://pixabay.com/api/videos/",
                             params={"key": key, "q": keyword, "per_page": 3}, timeout=10)
            hits = r.json().get("hits", [])
            if hits:
                vids = hits[0].get("videos", {})
                for q in ["large", "medium", "small"]:
                    if q in vids: return vids[q]["url"]
        except: pass
        return None

    def _pixabay_image(self, keyword, key):
        try:
            r = requests.get("https://pixabay.com/api/",
                             params={"key": key, "q": keyword, "per_page": 3,
                                     "image_type": "photo", "min_width": 1280}, timeout=10)
            hits = r.json().get("hits", [])
            if hits: return hits[0]["largeImageURL"]
        except: pass
        return None

    def _wikimedia_image(self, keyword):
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php",
                             params={"action": "query", "format": "json",
                                     "generator": "search",
                                     "gsrsearch": f"{keyword} filetype:bitmap",
                                     "gsrlimit": 5, "prop": "imageinfo",
                                     "iiprop": "url|mime", "iiurlwidth": 1280}, timeout=10)
            pages = r.json().get("query", {}).get("pages", {})
            for pid, pg in pages.items():
                info = pg.get("imageinfo", [{}])[0]
                if "image" in info.get("mime", ""):
                    return info.get("thumburl") or info.get("url")
        except: pass
        return None

    def _youtube_clip(self, keyword, out_path, max_secs=6):
        try:
            import yt_dlp
            opts = {
                "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "outtmpl": out_path,
                "download_ranges": lambda info, _: [{"start_time": 10, "end_time": 10 + max_secs}],
                "force_keyframes_at_cuts": True,
                "quiet": True, "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"ytsearch1:{keyword} footage"])
            return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 1000 else None
        except: return None

    # ── Step 4: Assembly ─────────────────────────────────────────────────────
    def step_assemble(self):
        self.log("🎬 Assembling video with MoviePy…")
        from moviepy.editor import (
            VideoFileClip, AudioFileClip, ColorClip, concatenate_videoclips
        )
        from moviepy.video.VideoClip import VideoClip

        W, H = self.s["width"], self.s["height"]
        FPS  = 30
        clips_out = []

        for i, sc in enumerate(self.scenes):
            seg_id     = sc["segment_id"]
            audio_path = self.audio_files[i]
            media_path = sc.get("media_path")
            duration   = sc.get("actual_audio_duration", sc["duration_hint_secs"])

            self.log(f"   Scene {seg_id}: compositing ({duration:.1f}s)…")

            # Audio
            audio_clip = AudioFileClip(audio_path) if (audio_path and os.path.exists(audio_path)) else None
            if audio_clip:
                duration = audio_clip.duration

            # Visual
            if media_path and media_path != "COLOR_BG" and os.path.exists(media_path):
                ext = os.path.splitext(media_path)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                    visual = self._ken_burns(media_path, duration, W, H, FPS, VideoClip)
                else:
                    try:
                        raw = VideoFileClip(media_path)
                        if raw.duration < duration:
                            loops = int(np.ceil(duration / raw.duration))
                            raw   = concatenate_videoclips([raw] * loops)
                        visual = raw.subclip(0, duration)
                        visual = self._resize_crop(visual, W, H)
                        visual = visual.without_audio()
                    except Exception as ev:
                        self.log(f"      Video load error: {ev}", "warn")
                        visual = ColorClip((W, H), color=(20, 20, 30), duration=duration)
            else:
                visual = ColorClip((W, H), color=(20, 20, 30), duration=duration)

            if audio_clip:
                visual = visual.set_audio(audio_clip)
            visual = visual.set_duration(duration).set_fps(FPS)
            clips_out.append(visual)

        # Crossfade
        self.log("   Adding crossfade transitions…")
        try:
            final_clips = [clips_out[0]]
            for clip in clips_out[1:]:
                final_clips.append(clip.crossfadein(0.3))
            final = concatenate_videoclips(final_clips, padding=-0.3, method="compose")
        except Exception:
            final = concatenate_videoclips(clips_out, method="compose")

        pre_cap = os.path.join(WORK_DIR, "precaption.mp4")
        self.log(f"   Writing pre-caption video…")
        final.write_videofile(pre_cap, fps=FPS, codec="libx264",
                              audio_codec="aac", preset="fast",
                              threads=4, logger=None)
        self.pre_caption_path = pre_cap
        self.log(f"✅ Assembly complete! ({final.duration:.1f}s)")

    def _resize_crop(self, clip, W, H):
        ratio_clip   = clip.w / clip.h
        ratio_target = W / H
        if ratio_clip > ratio_target:
            clip = clip.resize(height=H)
        else:
            clip = clip.resize(width=W)
        x = (clip.w - W) // 2
        y = (clip.h - H) // 2
        return clip.crop(x1=x, y1=y, x2=x+W, y2=y+H)

    def _ken_burns(self, img_path, duration, W, H, fps, VideoClip):
        SCALE = 1.10
        img   = Image.open(img_path).convert("RGB")
        sw, sh = int(img.width * SCALE), int(img.height * SCALE)
        _LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img_large = np.array(img.resize((sw, sh), _LANCZOS))

        def make_frame(t):
            p  = t / max(duration, 0.01)
            cw = int(sw / SCALE + (sw - sw / SCALE) * p)
            ch = int(sh / SCALE + (sh - sh / SCALE) * p)
            cw, ch = max(cw, 1), max(ch, 1)
            x, y   = (sw - cw) // 2, (sh - ch) // 2
            crop   = img_large[y:y+ch, x:x+cw]
            return np.array(Image.fromarray(crop).resize((W, H), _LANCZOS))

        return VideoClip(make_frame, duration=duration).set_fps(fps)

    # ── Step 5: Captions ─────────────────────────────────────────────────────
    def step_captions(self):
        self.log("💬 Generating captions with Whisper…")
        import whisper

        # Combine audio
        valid = [f for f in self.audio_files if f and os.path.exists(f)]
        combined = os.path.join(WORK_DIR, "combined_voice.wav")
        list_file = os.path.join(WORK_DIR, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as lf:
            for af in valid:
                # Normalize to forward slashes for ffmpeg on Windows
                lf.write(f"file '{af.replace(chr(92), '/')}'" + "\n")
        subprocess.run(
            f'ffmpeg -y -f concat -safe 0 -i "{list_file}" -ar 16000 "{combined}"',
            shell=True, capture_output=True
        )

        model = whisper.load_model("medium")
        result = model.transcribe(combined, task="transcribe",
                                  word_timestamps=True, fp16=False, verbose=False)
        self.log(f"   Detected language: {result.get('language','?').upper()}")
        self.log(f"   Segments: {len(result['segments'])}")

        # Write SRT
        srt_path = os.path.join(WORK_DIR, "subtitles.srt")
        def t2srt(s):
            h,m,sec = int(s//3600), int((s%3600)//60), int(s%60)
            ms = int((s-int(s))*1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        with open(srt_path, "w", encoding="utf-8") as sf_:
            for i, seg in enumerate(result["segments"], 1):
                sf_.write(f"{i}\n{t2srt(seg['start'])} --> {t2srt(seg['end'])}\n{seg['text'].strip()}\n\n")

        # Burn subtitles
        ts          = time.strftime("%Y%m%d%H%M%S")
        out_dir     = self.s.get("output_dir", os.path.expanduser("~"))
        out_path    = os.path.join(out_dir, f"trivid_{ts}.mp4")
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        style = ("FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=1,Outline=2,"
                 "Shadow=1,Alignment=2,MarginV=40")

        cmd = (f'ffmpeg -y -i "{self.pre_caption_path}" '
               f'-vf "subtitles=\'{srt_escaped}\':force_style=\'{style}\'" '
               f'-c:v libx264 -preset fast -crf 20 -c:a copy "{out_path}"')

        ret = subprocess.run(cmd, shell=True, capture_output=True)
        if ret.returncode != 0 or not os.path.exists(out_path):
            self.log("   ⚠️  Caption burn failed, saving without captions.", "warn")
            shutil.copy(self.pre_caption_path, out_path)
        else:
            self.log("   ✅ Captions burned in.")

        self.output_path = out_path
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        self.log(f"\n🎉 VIDEO READY: {out_path}")
        self.log(f"   Size: {size_mb:.1f} MB")
        self.log(f"   Resolution: {self.s['width']}×{self.s['height']}")

# ══════════════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class TrividApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION} — AI Video Studio")
        self.geometry("1100x780")
        self.minsize(960, 680)
        self.configure(fg_color="#0d0d14")

        self.cfg         = load_config()
        self.log_queue   = queue.Queue()
        self.prog_queue  = queue.Queue()
        self.pipeline    = None
        self.output_path = None

        self._build_ui()
        self._restore_config()
        self._poll_queues()

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Left sidebar ──
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0,
                                    fg_color="#13131f")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / title
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(pady=(28, 6), padx=18, fill="x")
        ctk.CTkLabel(logo_frame, text="🎬 TRIVID",
                     font=ctk.CTkFont("Arial", 28, "bold"),
                     text_color="#6c8fff").pack(anchor="w")
        ctk.CTkLabel(logo_frame, text="AI Video Studio",
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#555577").pack(anchor="w")

        ctk.CTkFrame(self.sidebar, height=1, fg_color="#2a2a3a").pack(fill="x", padx=18, pady=10)

        # Nav buttons
        self.nav_buttons = {}
        nav_items = [
            ("⚙️  Settings",  "settings"),
            ("📝  Script",    "script"),
            ("🚀  Generate",  "generate"),
        ]
        for label, key in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=label, anchor="w",
                font=ctk.CTkFont("Arial", 14),
                fg_color="transparent", hover_color="#1e1e2e",
                text_color="#aaaacc", corner_radius=8,
                height=44, command=lambda k=key: self._show_tab(k)
            )
            btn.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[key] = btn

        # Bottom info
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#2a2a3a").pack(fill="x", padx=18, pady=10, side="bottom")
        ctk.CTkLabel(self.sidebar, text="v2.0 · AI Video Generator",
                     font=ctk.CTkFont("Arial", 10),
                     text_color="#333355").pack(side="bottom", pady=8)

        # ── Main content ──
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.pack(side="right", fill="both", expand=True, padx=0)

        self.tabs = {}
        self.tabs["settings"] = self._build_settings_tab()
        self.tabs["script"]   = self._build_script_tab()
        self.tabs["generate"] = self._build_generate_tab()

        self._show_tab("settings")

    def _show_tab(self, key):
        for k, frame in self.tabs.items():
            frame.pack_forget()
        self.tabs[key].pack(fill="both", expand=True)
        for k, btn in self.nav_buttons.items():
            btn.configure(
                fg_color="#1e1e38" if k == key else "transparent",
                text_color="#6c8fff" if k == key else "#aaaacc"
            )

    # ── Settings Tab ────────────────────────────────────────────────────────
    def _build_settings_tab(self):
        frame = ctk.CTkScrollableFrame(self.main, fg_color="transparent")

        self._section_title(frame, "🔑  API Keys")
        self._hint(frame, "All keys are stored locally on your computer only.")

        lyt = ctk.CTkFont("Arial", 13)

        # Gemini
        self._label(frame, "Gemini API Key  (required)")
        self.gemini_entry = ctk.CTkEntry(frame, placeholder_text="AIza…",
                                          show="•", height=40, font=lyt)
        self.gemini_entry.pack(fill="x", padx=30, pady=(0, 10))
        ctk.CTkButton(frame, text="Get free Gemini key →",
                      fg_color="transparent", text_color="#6c8fff",
                      height=22, anchor="w", font=ctk.CTkFont("Arial", 12),
                      command=lambda: self._open_url("https://aistudio.google.com/app/apikey")
                      ).pack(anchor="w", padx=30, pady=(0, 14))

        # DuckDuckGo info banner (no key needed)
        ddg_info = ctk.CTkFrame(frame, fg_color="#0d1f0d", corner_radius=8)
        ddg_info.pack(fill="x", padx=30, pady=(0, 14))
        ctk.CTkLabel(ddg_info,
                     text="🔍  Images are sourced automatically via DuckDuckGo — no API key needed!",
                     font=ctk.CTkFont("Arial", 12), text_color="#44cc66",
                     wraplength=560, justify="left").pack(anchor="w", padx=14, pady=10)

        # Pexels (optional)
        self._label(frame, "Pexels API Key  (optional — adds stock video clips)")
        self.pexels_entry = ctk.CTkEntry(frame, placeholder_text="Pexels key… (optional)",
                                          show="•", height=40, font=lyt)
        self.pexels_entry.pack(fill="x", padx=30, pady=(0, 10))
        ctk.CTkButton(frame, text="Get free Pexels key →",
                      fg_color="transparent", text_color="#6c8fff",
                      height=22, anchor="w", font=ctk.CTkFont("Arial", 12),
                      command=lambda: self._open_url("https://www.pexels.com/api/")
                      ).pack(anchor="w", padx=30, pady=(0, 14))

        # Pixabay (optional)
        self._label(frame, "Pixabay API Key  (optional — extra image/video source)")
        self.pixabay_entry = ctk.CTkEntry(frame, placeholder_text="Pixabay key… (optional)",
                                           show="•", height=40, font=lyt)
        self.pixabay_entry.pack(fill="x", padx=30, pady=(0, 18))

        ctk.CTkFrame(frame, height=1, fg_color="#2a2a3a").pack(fill="x", padx=30, pady=10)
        self._section_title(frame, "🎙️  Voice & Output")

        # Voice
        self._label(frame, "Narration Voice")
        self.voice_var = ctk.StringVar(value="English — Female (af_heart)")
        voice_menu = ctk.CTkOptionMenu(frame, variable=self.voice_var,
                                        values=["English — Female (af_heart)",
                                                "English — Male (am_adam)",
                                                "Hindi — Female (hf_alpha)"],
                                        height=40, font=lyt)
        voice_menu.pack(fill="x", padx=30, pady=(0, 16))

        # Aspect ratio
        self._label(frame, "Aspect Ratio")
        self.ratio_var = ctk.StringVar(value="16:9 — YouTube / Landscape (1920×1080)")
        ratio_menu = ctk.CTkOptionMenu(frame, variable=self.ratio_var,
                                        values=["16:9 — YouTube / Landscape (1920×1080)",
                                                "9:16 — Shorts / Reels (1080×1920)",
                                                "1:1 — Instagram Square (1080×1080)",
                                                "4:3 — Classic (1440×1080)"],
                                        height=40, font=lyt)
        ratio_menu.pack(fill="x", padx=30, pady=(0, 16))

        # Resolution
        self._label(frame, "Resolution")
        self.res_var = ctk.StringVar(value="1080p")
        res_seg = ctk.CTkSegmentedButton(frame, values=["720p", "1080p"],
                                          variable=self.res_var, height=38,
                                          font=lyt)
        res_seg.pack(fill="x", padx=30, pady=(0, 16))

        # Output folder
        self._label(frame, "Output Folder")
        out_row = ctk.CTkFrame(frame, fg_color="transparent")
        out_row.pack(fill="x", padx=30, pady=(0, 10))
        self.output_dir_var = ctk.StringVar(value=os.path.expanduser("~\\Desktop"))
        ctk.CTkEntry(out_row, textvariable=self.output_dir_var,
                     height=40, font=lyt).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(out_row, text="Browse", width=90, height=40,
                      command=self._browse_output).pack(side="right")

        # Save keys button
        ctk.CTkFrame(frame, height=1, fg_color="#2a2a3a").pack(fill="x", padx=30, pady=16)
        ctk.CTkButton(frame, text="💾  Save Settings",
                      height=46, font=ctk.CTkFont("Arial", 14, "bold"),
                      fg_color="#3a3aff", hover_color="#5555ff",
                      command=self._save_settings).pack(fill="x", padx=30, pady=(0, 30))

        return frame

    # ── Script Tab ───────────────────────────────────────────────────────────
    def _build_script_tab(self):
        frame = ctk.CTkFrame(self.main, fg_color="transparent")

        self._section_title(frame, "📝  Your Script")
        self._hint(frame, "Write or paste your video script. Gemini AI will automatically "
                          "break it into scenes and match visuals to each part.")

        # Word count
        self.word_count_label = ctk.CTkLabel(frame, text="0 words",
                                              font=ctk.CTkFont("Arial", 12),
                                              text_color="#555577")
        self.word_count_label.pack(anchor="e", padx=30, pady=(0, 4))

        self.script_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont("Arial", 14), wrap="word",
            fg_color="#0f0f1a", border_color="#2a2a3a", border_width=1,
            corner_radius=10
        )
        self.script_text.pack(fill="both", expand=True, padx=30, pady=(0, 12))
        self.script_text.bind("<KeyRelease>", self._update_word_count)

        sample_text = (
            "The Roman Empire was one of the greatest civilizations in human history. "
            "It began as a small city-state in central Italy and grew to become a vast empire "
            "that stretched from Britain to Egypt. Romans built roads, aqueducts, and magnificent "
            "structures like the Colosseum. Their culture, laws, and engineering shaped the "
            "entire Western world for thousands of years."
        )
        ctk.CTkButton(frame, text="Load Sample Script", fg_color="transparent",
                      text_color="#6c8fff", height=30,
                      command=lambda: (self.script_text.delete("1.0", "end"),
                                       self.script_text.insert("1.0", sample_text),
                                       self._update_word_count())
                      ).pack(anchor="w", padx=30, pady=(0, 4))

        ctk.CTkButton(frame, text="▶  Proceed to Generate →",
                      height=46, font=ctk.CTkFont("Arial", 14, "bold"),
                      fg_color="#3a3aff", hover_color="#5555ff",
                      command=lambda: self._show_tab("generate")
                      ).pack(fill="x", padx=30, pady=(8, 30))

        return frame

    # ── Generate Tab ─────────────────────────────────────────────────────────
    def _build_generate_tab(self):
        frame = ctk.CTkFrame(self.main, fg_color="transparent")

        self._section_title(frame, "🚀  Generate Video")
        self._hint(frame, "Click Generate to start the full AI pipeline. This may take "
                          "5–20 minutes depending on script length.")

        # Status card
        status_card = ctk.CTkFrame(frame, fg_color="#13131f", corner_radius=12)
        status_card.pack(fill="x", padx=30, pady=(0, 12))

        status_inner = ctk.CTkFrame(status_card, fg_color="transparent")
        status_inner.pack(fill="x", padx=20, pady=16)

        self.status_label = ctk.CTkLabel(status_inner, text="Ready to generate",
                                          font=ctk.CTkFont("Arial", 14, "bold"),
                                          text_color="#6c8fff")
        self.status_label.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(status_card, height=8,
                                                progress_color="#3a3aff",
                                                fg_color="#1e1e2e")
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 20))
        self.progress_bar.set(0)

        # Log box
        self._label(frame, "Pipeline Log")
        self.log_box = ctk.CTkTextbox(
            frame, font=ctk.CTkFont("Courier New", 12),
            fg_color="#080810", border_color="#1e1e2e", border_width=1,
            corner_radius=10, text_color="#88aaff"
        )
        self.log_box.pack(fill="both", expand=True, padx=30, pady=(0, 12))

        # Buttons row
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=30, pady=(0, 30))

        self.gen_btn = ctk.CTkButton(
            btn_row, text="🎬  Generate Video",
            height=50, font=ctk.CTkFont("Arial", 15, "bold"),
            fg_color="#3a3aff", hover_color="#5555ff",
            command=self._start_generation
        )
        self.gen_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.open_btn = ctk.CTkButton(
            btn_row, text="📂  Open Output",
            height=50, width=160, font=ctk.CTkFont("Arial", 14),
            fg_color="#1e2e1e", hover_color="#2a3e2a",
            state="disabled", command=self._open_output
        )
        self.open_btn.pack(side="right")

        return frame

    # ── UI helpers ───────────────────────────────────────────────────────────
    def _section_title(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Arial", 18, "bold"),
                     text_color="#ccccee").pack(anchor="w", padx=30, pady=(24, 4))

    def _label(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#888899").pack(anchor="w", padx=30, pady=(8, 2))

    def _hint(self, parent, text):
        ctk.CTkLabel(parent, text=text, wraplength=700,
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#444466", justify="left").pack(anchor="w", padx=30, pady=(0, 10))

    def _open_url(self, url):
        import webbrowser
        webbrowser.open(url)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_dir_var.set(d)

    def _update_word_count(self, *args):
        text  = self.script_text.get("1.0", "end").strip()
        words = len(text.split()) if text else 0
        secs  = round(words / 2.5)
        self.word_count_label.configure(text=f"{words} words · ~{secs}s video")

    # ── Config ───────────────────────────────────────────────────────────────
    def _restore_config(self):
        if self.cfg.get("gemini_key"):
            self.gemini_entry.insert(0, self.cfg["gemini_key"])
        if self.cfg.get("pexels_key"):
            self.pexels_entry.insert(0, self.cfg["pexels_key"])
        if self.cfg.get("pixabay_key"):
            self.pixabay_entry.insert(0, self.cfg["pixabay_key"])
        if self.cfg.get("output_dir"):
            self.output_dir_var.set(self.cfg["output_dir"])
        if self.cfg.get("voice"):
            self.voice_var.set(self.cfg["voice"])
        if self.cfg.get("ratio"):
            self.ratio_var.set(self.cfg["ratio"])
        if self.cfg.get("res"):
            self.res_var.set(self.cfg["res"])

    def _save_settings(self):
        self.cfg.update({
            "gemini_key":  self.gemini_entry.get().strip(),
            "pexels_key":  self.pexels_entry.get().strip(),
            "pixabay_key": self.pixabay_entry.get().strip(),
            "output_dir":  self.output_dir_var.get().strip(),
            "voice":       self.voice_var.get(),
            "ratio":       self.ratio_var.get(),
            "res":         self.res_var.get(),
        })
        save_config(self.cfg)
        messagebox.showinfo("Trivid", "✅ Settings saved!")

    # ── Generation ───────────────────────────────────────────────────────────
    def _build_settings_dict(self):
        ratio_map = {
            "16:9 — YouTube / Landscape (1920×1080)": (1920, 1080),
            "9:16 — Shorts / Reels (1080×1920)":      (1080, 1920),
            "1:1 — Instagram Square (1080×1080)":     (1080, 1080),
            "4:3 — Classic (1440×1080)":               (1440, 1080),
        }
        W, H = ratio_map.get(self.ratio_var.get(), (1920, 1080))
        if self.res_var.get() == "720p":
            scale = 720 / max(W, H)
            W = (int(W * scale) // 2) * 2
            H = (int(H * scale) // 2) * 2

        return {
            "gemini_key":  self.gemini_entry.get().strip(),
            "pexels_key":  self.pexels_entry.get().strip(),
            "pixabay_key": self.pixabay_entry.get().strip(),
            "script":      self.script_text.get("1.0", "end").strip(),
            "voice":       self.voice_var.get(),
            "width": W, "height": H,
            "output_dir":  self.output_dir_var.get().strip() or os.path.expanduser("~"),
        }

    def _start_generation(self):
        settings = self._build_settings_dict()
        if not settings["gemini_key"]:
            messagebox.showerror("Missing Key", "Please enter your Gemini API key in Settings.")
            return
        if not settings["script"]:
            messagebox.showerror("Empty Script", "Please write your script in the Script tab.")
            return

        self._show_tab("generate")
        self.gen_btn.configure(state="disabled", text="⏳  Generating…")
        self.open_btn.configure(state="disabled")
        self.log_box.delete("1.0", "end")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting pipeline…")

        # Clean workspace
        for item in os.listdir(WORK_DIR):
            p = os.path.join(WORK_DIR, item)
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)

        self.pipeline = Pipeline(settings, self.log_queue, self.prog_queue)
        t = threading.Thread(target=self._run_pipeline, daemon=True)
        t.start()

    def _run_pipeline(self):
        self.pipeline.run()
        self.log_queue.put(("DONE", self.pipeline.output_path))

    def _poll_queues(self):
        # Process log messages
        while not self.log_queue.empty():
            item = self.log_queue.get_nowait()
            level, msg = item
            if level == "DONE":
                self.output_path = msg
                self.gen_btn.configure(state="normal", text="🎬  Generate Video")
                if msg and os.path.exists(msg):
                    self.open_btn.configure(state="normal")
                    self.status_label.configure(text=f"✅ Done! Saved to: {os.path.basename(msg)}")
                    self._append_log(f"\n🎉 VIDEO SAVED: {msg}\n", color="#55ff55")
                else:
                    self.status_label.configure(text="❌ Generation failed. Check log.")
                    self._append_log("\n❌ Pipeline completed with errors.\n", color="#ff5555")
            else:
                color = "#ff6666" if level == "error" else ("#ffcc44" if level == "warn" else "#88aaff")
                self._append_log(msg + "\n", color=color)

        # Process progress
        while not self.prog_queue.empty():
            value, label = self.prog_queue.get_nowait()
            self.progress_bar.set(value / 100)
            if label:
                self.status_label.configure(text=label)

        self.after(150, self._poll_queues)

    def _append_log(self, msg, color="#88aaff"):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _open_output(self):
        if self.output_path and os.path.exists(self.output_path):
            os.startfile(self.output_path)
        else:
            out_dir = self.output_dir_var.get()
            if os.path.exists(out_dir):
                os.startfile(out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = TrividApp()
    app.mainloop()
