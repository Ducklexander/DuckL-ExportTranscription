# -*- coding: utf-8 -*-
"""
ExportTranscription 核心：語音 → 有說話人標記、有標點、分段的逐字稿。

流程
  1. 解碼：PyAV 把 wav/mp3/mp4/m4a… 轉成 16 kHz 單聲道（不需要另外裝 ffmpeg）
  2. 說話人分離：pyannote segmentation 3.0 + 3D-Speaker CAM++ 聲紋（sherpa-onnx），
     先刻意切細，再用聲紋質心做第二次聚類，自動判斷人數
  3. 語音辨識：Breeze-ASR-25（聯發科針對台灣華語、中英夾雜微調的 Whisper），
     CTranslate2 int8，逐字時間戳
  4. 把每個字對到說話人，同一人連續的話合成一段
  5. 標點：CT-Transformer（中英），再用停頓補逗號
  6. 輸出 .txt（逐字稿）與 .srt（字幕）
"""
import os, re, sys, time, math, datetime

import numpy as np

SR = 16000


class Cancelled(Exception):
    pass


# 介面訊息（逐字稿本身的語言另外由辨識結果決定）
MSG = {
    "zh": {
        "dec": "讀取音訊", "dia": "分辨說話人", "asr": "語音辨識", "fmt": "整理段落與標點",
        "done": "完成", "length": "  長度 {t}", "speakers": "  說話人 {n} 位（{s:.0f}s）",
        "device": "  辨識裝置：{d}", "finished": "  完成，共 {n} 段（{s:.0f}s）",
        "too_short": "音訊太短或沒有聲音", "no_speech": "沒有辨識到任何語音",
        "gpu_missing": "  偵測到 NVIDIA 顯卡，但還沒安裝 GPU 加速，改用 CPU"
                       "（按視窗上的「安裝 GPU 加速」可快約 5 倍）",
        "gpu_fail": "  GPU 無法使用，改用 CPU（{e}）",
    },
    "en": {
        "dec": "Reading audio", "dia": "Finding speakers", "asr": "Recognising speech",
        "fmt": "Punctuation and paragraphs",
        "done": "Done", "length": "  Length {t}", "speakers": "  {n} speaker(s) ({s:.0f}s)",
        "device": "  Running on: {d}", "finished": "  Done, {n} paragraphs ({s:.0f}s)",
        "too_short": "The audio is too short or silent", "no_speech": "No speech was recognised",
        "gpu_missing": "  NVIDIA GPU found but GPU acceleration is not installed, using CPU "
                       "(click \"Install GPU acceleration\" for about 5x speed)",
        "gpu_fail": "  GPU unavailable, using CPU ({e})",
    },
}


def gpu_state(app=None):
    """'ok' = 有 NVIDIA 顯卡且已裝 cuBLAS；'missing' = 有顯卡沒裝；'none' = 沒有顯卡。"""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return "none"
    except Exception:
        return "none"
    # 用實際載入來判斷：程式資料夾或系統 PATH 裡有 cuBLAS 都算
    try:
        import ctypes
        for f in GPU_DLLS:
            ctypes.WinDLL(os.path.join(app or app_dir(), f)
                          if os.path.exists(os.path.join(app or app_dir(), f)) else f)
        return "ok"
    except OSError:
        return "missing"


GPU_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")
# 先從 NVIDIA 官方 CDN 下載（快），失敗再用 GitHub Release 上的備份
GPU_PACK_URLS = (
    "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/"
    "libcublas-windows-x86_64-12.8.4.1-archive.zip",
    "https://github.com/Ducklexander/DuckL-ExportTranscription/"
    "releases/latest/download/DuckL-ExportTranscription_GPU.zip",
)


def install_gpu(progress=None, cancel=None, urls=GPU_PACK_URLS, app=None):
    """下載 cuBLAS 並把需要的 DLL 解壓到程式資料夾。progress(已下載, 總大小)。"""
    import urllib.request, zipfile, shutil
    d = app or app_dir()
    tmp = os.path.join(d, "_gpu_download.zip")
    err = None
    try:
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DuckL-ExportTranscription"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    got = 0
                    while True:
                        if cancel and cancel.is_set():
                            raise Cancelled()
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if progress:
                            progress(got, total)
                with zipfile.ZipFile(tmp) as z:
                    members = [i for i in z.infolist()
                               if os.path.basename(i.filename) in GPU_DLLS]
                    if len(members) < len(GPU_DLLS):
                        raise RuntimeError("cuBLAS DLL not found in download")
                    for info in members:
                        out = os.path.join(d, os.path.basename(info.filename))
                        with z.open(info) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
                return
            except Cancelled:
                raise
            except Exception as e:          # 換下一個來源
                err = e
        raise err
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def app_dir():
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def find_models(base=None):
    """回傳各模型路徑；找不到就丟出清楚的錯誤。"""
    base = base or os.path.join(app_dir(), "models")
    m = {
        "asr": os.path.join(base, "asr"),
        "seg": os.path.join(base, "diarization", "segmentation.onnx"),
        "spk": os.path.join(base, "diarization", "speaker.onnx"),
        "punct": os.path.join(base, "punct", "model.int8.onnx"),
    }
    missing = [p for k, p in m.items()
               if not os.path.exists(os.path.join(p, "model.bin") if k == "asr" else p)]
    if missing:
        raise FileNotFoundError("找不到模型檔 / model files missing:\n" + "\n".join(missing))
    return m


def default_threads():
    n = os.cpu_count() or 4
    return max(2, min(8, n // 2 if n >= 8 else n))


# ------------------------------------------------------------------ 解碼
def load_audio(path):
    from faster_whisper.audio import decode_audio
    a = decode_audio(path, sampling_rate=SR)
    return np.ascontiguousarray(a, dtype=np.float32)


# ------------------------------------------------------------------ 說話人
def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Diarizer:
    def __init__(self, seg_model, spk_model, threads):
        import sherpa_onnx as so
        self.so = so
        self.seg_model, self.spk_model, self.threads = seg_model, spk_model, threads
        self.ext = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
            model=spk_model, num_threads=threads))

    def embed(self, audio, a, b):
        st = self.ext.create_stream()
        st.accept_waveform(SR, audio[int(a * SR):int(b * SR)])
        st.input_finished()
        return np.asarray(self.ext.compute(st), dtype=np.float32)

    def run(self, audio, num_speakers=0, progress=None, cancel=None):
        """回傳 [(start, end, speaker_index)]，speaker 依第一次出現排序 0,1,2…"""
        so = self.so
        # 第一階段：門檻偏低，寧可把一個人切成好幾群，也不要把兩個人併成一群
        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=self.seg_model),
                num_threads=self.threads),
            embedding=so.SpeakerEmbeddingExtractorConfig(
                model=self.spk_model, num_threads=self.threads),
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=0.7),
            min_duration_on=0.3, min_duration_off=0.5)
        d = so.OfflineSpeakerDiarization(cfg)

        def cb(done, total):
            if progress and total:
                progress(0.85 * done / total)
            return 1 if (cancel and cancel.is_set()) else 0

        res = d.process(audio, callback=cb).sort_by_start_time()
        if cancel and cancel.is_set():
            raise Cancelled()
        turns = [(s.start, s.end, s.speaker) for s in res]
        if not turns:
            return []

        # 第二階段：每一段算聲紋，依群算加權質心，再做階層式合併
        embs = {}
        for i, (a, b, _) in enumerate(turns):
            if b - a >= 0.8:
                embs[i] = _norm(self.embed(audio, a, min(b, a + 20)))
            if progress and i % 20 == 0:
                progress(0.85 + 0.13 * i / len(turns))
            if cancel and cancel.is_set():
                raise Cancelled()
        out = self.recluster(turns, embs, num_speakers)
        if progress:
            progress(1.0)
        return out

    def recluster(self, turns, embs, num_speakers=0):
        labels = sorted({t[2] for t in turns})
        dur = {l: 0.0 for l in labels}
        acc = {l: None for l in labels}
        for i, (a, b, l) in enumerate(turns):
            dur[l] += b - a
            if i in embs:
                w = min(b - a, 20.0)
                acc[l] = embs[i] * w if acc[l] is None else acc[l] + embs[i] * w
        # 沒有任何夠長片段的群：沒有聲紋可比，併到時間上最近的群
        clusters = {l: {"members": {l}, "dur": dur[l], "c": _norm(acc[l])}
                    for l in labels if acc[l] is not None}
        if not clusters:
            return [(a, b, 0) for a, b, _ in turns]

        total = sum(c["dur"] for c in clusters.values())

        def merge(x, y):
            cx, cy = clusters[x], clusters.pop(y)
            cx["c"] = _norm(cx["c"] * cx["dur"] + cy["c"] * cy["dur"])
            cx["dur"] += cy["dur"]
            cx["members"] |= cy["members"]

        def best_pair(pool):
            best, bp = -2.0, None
            keys = list(pool)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    s = float(clusters[keys[i]]["c"] @ clusters[keys[j]]["c"])
                    if s > best:
                        best, bp = s, (keys[i], keys[j])
            return best, bp

        if num_speakers and num_speakers > 0:
            while len(clusters) > num_speakers:
                _, (x, y) = best_pair(clusters)
                if clusters[x]["dur"] < clusters[y]["dur"]:
                    x, y = y, x
                merge(x, y)
        else:
            # 自動：聲紋質心夠像就合併
            while len(clusters) > 1:
                s, (x, y) = best_pair(clusters)
                if s < self.MERGE_SIM:
                    break
                if clusters[x]["dur"] < clusters[y]["dur"]:
                    x, y = y, x
                merge(x, y)
            # 講話太少的群（通常是笑聲、雜音、重疊講話）併進最像的大群
            small = max(8.0, 0.03 * total)
            while len(clusters) > 1:
                tiny = [k for k, c in clusters.items() if c["dur"] < small]
                if not tiny:
                    break
                k = min(tiny, key=lambda k: clusters[k]["dur"])
                big = [j for j in clusters if j != k]
                j = max(big, key=lambda j: float(clusters[j]["c"] @ clusters[k]["c"]))
                merge(j, k)

        owner = {}
        for k, c in clusters.items():
            for m in c["members"]:
                owner[m] = k
        cents = {k: c["c"] for k, c in clusters.items()}

        out = []
        for i, (a, b, l) in enumerate(turns):
            k = owner.get(l)
            if k is None or (i in embs and b - a >= 2.0):
                # 長片段直接比對最終質心，修正第一階段的誤分；沒聲紋的群用最近片段
                if i in embs:
                    sims = {kk: float(embs[i] @ c) for kk, c in cents.items()}
                    kb = max(sims, key=sims.get)
                    if k is None or sims[kb] - sims[k] > 0.1:
                        k = kb
                if k is None:
                    k = next(iter(cents))
            out.append([a, b, k])

        # 依第一次開口的順序編號 A, B, C…
        order = {}
        for a, b, k in out:
            if k not in order:
                order[k] = len(order)
        return [(a, b, order[k]) for a, b, k in out]

    MERGE_SIM = 0.72


# ------------------------------------------------------------------ 辨識
HALLUCINATIONS = re.compile(
    r"(字幕|訂閱|点赞|點讚|點贊|按讚|感謝觀看|謝謝觀看|謝謝收看|請不吝|Amara|中文字幕|"
    r"小鈴鐺|明鏡|優優獨播|YoYo|Subtitles? by|Thank you for watching)", re.I)


class Recognizer:
    def __init__(self, asr_dir, threads, device="auto", log=None, msg=None):
        msg = msg or MSG["zh"]
        from faster_whisper import WhisperModel
        self.device = "cpu"
        self.model = None
        state = gpu_state() if device in ("auto", "cuda") else "none"
        if state == "missing" and log:
            # 沒有 cuBLAS 就不要試：失敗後同一個程式裡再試會卡住
            log(msg["gpu_missing"])
        if state == "ok":
            try:
                m = WhisperModel(asr_dir, device="cuda", compute_type="int8_float16")
                # 先跑一小段確認顯卡真的能用
                list(m.transcribe(np.zeros(SR, np.float32), language="zh")[0])
                self.model, self.device = m, "cuda"
            except Exception as e:
                if log:
                    log(msg["gpu_missing"] if "cublas" in str(e).lower()
                        else msg["gpu_fail"].format(e=type(e).__name__))
        if self.model is None:
            self.model = WhisperModel(asr_dir, device="cpu", compute_type="int8",
                                      cpu_threads=threads)

    def run(self, audio, language="zh", hotwords="", progress=None, cancel=None):
        """回傳 ([(start, end, text, [(ws, we, word)…])…], 語言)"""
        total = len(audio) / SR
        kw = dict(beam_size=5, vad_filter=True,
                  vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                  word_timestamps=True, condition_on_previous_text=False,
                  language=None if language == "auto" else language)
        hw = " ".join(hotwords.replace("，", " ").replace(",", " ").split())
        if hw:
            kw["hotwords"] = hw
        segs, info = self.model.transcribe(audio, **kw)
        out = []
        for s in segs:
            if cancel and cancel.is_set():
                raise Cancelled()
            if progress:
                progress(min(1.0, s.end / total))
            # Whisper 在靜音或音樂段常會幻想出「字幕由…提供」「請訂閱」之類的句子
            if len(s.text) < 30 and HALLUCINATIONS.search(s.text) and s.avg_logprob < -0.4:
                continue
            if s.no_speech_prob > 0.8 and s.avg_logprob < -1.0:
                continue
            ws = [(w.start, w.end, w.word) for w in (s.words or []) if w.word.strip()]
            if ws and s.text.strip():
                out.append((s.start, s.end, s.text.strip(), ws))
        return out, info.language


# ------------------------------------------------------------------ 標點
PUNCT_ZH = "，。？！、；：,.?!;:…"


class Punctuator:
    def __init__(self, model, threads):
        import sherpa_onnx as so
        self.p = so.OfflinePunctuation(so.OfflinePunctuationConfig(
            model=so.OfflinePunctuationModelConfig(ct_transformer=model,
                                                   num_threads=min(threads, 4))))

    def run(self, texts, english=False, final=True):
        """texts = 同一人連續講的幾句（辨識器切好的語句）。回傳加好標點的字串。
        模型沒在語句交界處斷句的話，補一個逗號（那裡本來就有停頓）。"""
        clean = [re.sub(r"[，。？！、；：,.?!;:…]", "", t).strip() for t in texts]
        joined = ""
        for t in clean:
            if joined and joined[-1].isascii() and joined[-1].isalnum() \
                    and t[:1].isascii() and t[:1].isalnum():
                joined += " "
            joined += t
        if not joined.strip():
            return joined
        out = self.p.add_punctuation(joined)
        breaks, pos = set(), 0
        for t in clean[:-1]:
            pos += _nchars(t)
            breaks.add(pos)
        res, n = [], 0
        for i, ch in enumerate(out):
            res.append(ch)
            if not ch.isspace() and ch not in PUNCT_ZH:
                n += 1
                j = i + 1
                while j < len(out) and out[j].isspace():
                    j += 1
                nxt = out[j] if j < len(out) else ""
                if n in breaks and nxt and nxt not in PUNCT_ZH:
                    res.append("，")
        s = "".join(res).rstrip()
        if s and s[-1] not in "。？！?!.…":
            s = s.rstrip("，、；：,;:") + ("。" if final else "，")
        if english:
            s = (s.replace("，", ", ").replace("。", ". ").replace("？", "? ")
                  .replace("！", "! ").replace("、", ", ").replace("；", "; ")
                  .replace("：", ": "))
            s = re.sub(r"\s+([,.?!;:])", r"\1", s)
            s = re.sub(r"(^|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
            s = re.sub(r"\s{2,}", " ", s).strip()
        return s


CJK = "㐀-鿿豈-﫿"


def tidy(s):
    """中英之間留一個半形空白、收掉重複語助詞、標點前不留空白。"""
    s = re.sub(rf"([{CJK}])([A-Za-z0-9])", r"\1 \2", s)
    s = re.sub(rf"([A-Za-z0-9%])([{CJK}])", r"\1 \2", s)
    s = re.sub(r"\s+([，。？！、；：])", r"\1", s)
    s = re.sub(r"([，。？！、；：])\s+", r"\1", s)
    s = re.sub(r"([，、])\1+", r"\1", s)
    s = re.sub(r"[，、]([。？！])", r"\1", s)
    s = re.sub(r"(.{1,4}?)\1{5,}", r"\1\1\1", s)          # 對對對對對對對 → 對對對
    return s.strip()


# ------------------------------------------------------------------ 對齊
def _overlaps(a, b, turns, starts):
    """[a,b] 與各說話人重疊的秒數；完全沒重疊時回傳最近的說話人。"""
    i0 = max(0, int(np.searchsorted(starts, a)) - 8)
    ov, near, nd = {}, None, 1e9
    for ta, tb, spk in turns[i0:]:
        if ta > b + 5:
            break
        o = min(b, tb) - max(a, ta)
        if o > 0:
            ov[spk] = ov.get(spk, 0.0) + o
        dd = 0 if o > 0 else min(abs(a - tb), abs(ta - b))
        if dd < nd:
            nd, near = dd, spk
    return ov, near


def assign_units(segs, turns):
    """以辨識出的語句為單位決定說話人（句子不會被切在中間）。
    只有長句、而且裡面明顯換人講時，才在字與字的停頓處切開。
    segs = [(start, end, text, [(ws, we, word)…])]，回傳 [{spk,start,end,text,words}]
    文字一律用整句的辨識結果；逐字資料只拿來算時間（英文的逐字切分偶爾會錯位）。"""
    if not turns:
        turns = [(0.0, 1e9, 0)]
    starts = np.array([t[0] for t in turns])
    units = []
    for seg in segs:
        ss, se, text, ws = seg if len(seg) == 4 else (seg[0], seg[1], None, seg[2])
        if not ws:
            continue
        joined = "".join(w[2] for w in ws)
        if text is None:
            text = joined
        a, b = ws[0][0], ws[-1][1]
        ov, near = _overlaps(a, b, turns, starts)
        tot = sum(ov.values())
        dom = max(ov, key=ov.get) if ov else near
        aligned = _plain(joined) == _plain(text)
        if not ov or b - a < 3.0 or ov[dom] / tot >= 0.75 or not aligned:
            units.append({"spk": dom, "start": a, "end": b, "text": text, "words": ws})
            continue
        # 逐字判斷，再把太短的片段併回前後
        lab = []
        for w in ws:
            o, n = _overlaps(w[0], w[1], turns, starts)
            lab.append(max(o, key=o.get) if o else n)
        runs = []
        for w, l in zip(ws, lab):
            if runs and runs[-1][0] == l:
                runs[-1][1].append(w)
            else:
                runs.append([l, [w]])
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, (l, rw) in enumerate(runs):
                if rw[-1][1] - rw[0][0] < 1.2:
                    j = i - 1 if i > 0 else i + 1
                    if i > 0 and i + 1 < len(runs):
                        # 併到停頓比較短的那一邊
                        gl = rw[0][0] - runs[i - 1][1][-1][1]
                        gr = runs[i + 1][1][0][0] - rw[-1][1]
                        j = i - 1 if gl <= gr else i + 1
                    if j < i:
                        runs[j][1].extend(rw)
                    else:
                        runs[j][1][:0] = rw
                    del runs[i]
                    changed = True
                    break
            k = 0
            while k + 1 < len(runs):
                if runs[k][0] == runs[k + 1][0]:
                    runs[k][1].extend(runs.pop(k + 1)[1])
                    changed = True
                else:
                    k += 1
        for l, rw in runs:
            units.append({"spk": l, "start": rw[0][0], "end": rw[-1][1],
                          "text": "".join(w[2] for w in rw), "words": rw})
    return units


def _plain(t):
    return re.sub(r"[\s，。？！、；：,.?!;:…\"'“”‘’()（）-]", "", t).lower()


def _join_words(words, english):
    s = ""
    for _, _, w in words:
        t = w if english else w.strip()
        if not english and s and s[-1].isascii() and s[-1].isalnum() \
                and t[:1].isascii() and t[:1].isalnum():
            s += " "
        s += t
    return re.sub(r"\s+", " ", s).strip()


def _split_back(out, lens):
    """把整段加好標點的字串，依每句原本的字數切回去（標點跟著前一句）。"""
    parts, cur, n, k = [], [], 0, 0
    target = lens[0] if lens else 0
    for ch in out:
        is_char = not ch.isspace() and ch not in PUNCT_ZH
        if is_char and n >= target and k < len(lens) - 1:
            parts.append("".join(cur))
            cur = []
            k += 1
            target += lens[k]
        cur.append(ch)
        if is_char:
            n += 1
    parts.append("".join(cur))
    while len(parts) < len(lens):
        parts.append("")
    return parts


def _nchars(t):
    return len(re.sub(r"[\s，。？！、；：,.?!;:…]", "", t))


def build_paragraphs(units, punct, english=False, max_chars=200):
    """同一人連續講的話合成一段；太長的段落在句尾再切開。
    回傳 [{spk,start,end,text,cues:[(start,end,text)]}]"""
    blocks = []
    for u in units:
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip() if english else             _join_words([(0, 0, w) for w in re.split(r"(\s+)", u["text"])], False)
        if not u["text"]:
            continue
        if blocks and blocks[-1][-1]["spk"] == u["spk"] and u["start"] - blocks[-1][-1]["end"] < 8.0:
            blocks[-1].append(u)
        else:
            blocks.append([u])

    paras = []
    for bl in blocks:
        # 標點模型一次吃太長會變慢也不準，大約 300 字一批
        texts = []
        chunk, clen = [], 0
        for i, u in enumerate(bl):
            chunk.append(u)
            clen += len(u["text"])
            last = i == len(bl) - 1
            gap = 0.0 if last else bl[i + 1]["start"] - u["end"]
            # 盡量在明顯停頓處分批，批次尾端才不會斷在句子中間
            if english:
                # Whisper 的英文本來就有標點與大小寫，不再經過中文標點模型
                t = u["text"]
                if last and t and t[-1] not in ".?!":
                    t += "."
                texts.append(t)
                chunk, clen = [], 0
            elif last or (clen > 250 and gap >= 0.5) or clen > 450:
                out = punct.run([c["text"] for c in chunk], english, final=last)
                texts += _split_back(out, [_nchars(c["text"]) for c in chunk])
                chunk, clen = [], 0
        cues = [(u["start"], u["end"], t.strip()) for u, t in zip(bl, texts) if t.strip()]
        # 太長就在句尾切段
        cur, clen = [], 0
        groups = []
        for c in cues:
            cur.append(c)
            clen += len(c[2])
            if clen >= max_chars and c[2][-1:] in "。？！?!.":
                groups.append(cur)
                cur, clen = [], 0
        if cur:
            groups.append(cur)
        for g in groups:
            sep = " " if english else ""
            text = sep.join(c[2] for c in g)
            text = text.strip() if english else tidy(text)
            if not text.strip("，。？！、 ,.?!"):
                continue
            paras.append({"spk": bl[0]["spk"], "start": g[0][0], "end": g[-1][1],
                          "text": text, "cues": g})
    return paras


def compose(segs, turns, lang, punct):
    """辨識結果 + 說話人片段 → 段落。回傳 (paras, 說話人數)"""
    units = assign_units(segs, turns)
    # 重新依第一次開口順序編號（有些說話人可能完全沒有對到字）
    remap = {}
    for u in units:
        if u["spk"] not in remap:
            remap[u["spk"]] = len(remap)
        u["spk"] = remap[u["spk"]]
    paras = build_paragraphs(units, punct, lang == "en")
    if lang in ("zh", "yue"):
        for p in paras:
            p["text"] = to_traditional(p["text"])
            p["cues"] = [(a, b, to_traditional(t)) for a, b, t in p["cues"]]
    return paras, max(1, len(remap))


# ------------------------------------------------------------------ 輸出
def spk_name(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def fmt_ts(t, srt=False):
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_txt(path, paras, src, duration, nspk, timestamps=True, english=False):
    L = []
    if english:
        L.append(f"Transcript  {os.path.basename(src)}")
        L.append(f"Length {fmt_ts(duration)} | Speakers: {nspk} | "
                 f"Created {datetime.datetime.now():%Y-%m-%d %H:%M}")
        L.append("(Automatic transcript. Please proofread names and terms.)")
    else:
        L.append(f"逐字稿　{os.path.basename(src)}")
        L.append(f"長度 {fmt_ts(duration)}｜說話人 {nspk} 位"
                 f"（{'、'.join(spk_name(i) for i in range(nspk))}）｜"
                 f"產生於 {datetime.datetime.now():%Y-%m-%d %H:%M}")
        L.append("（自動辨識結果，人名與專有名詞請再校對）")
    L.append("─" * 30)
    L.append("")
    colon = ": " if english else "："
    for p in paras:
        head = f"[{fmt_ts(p['start'])}] " if timestamps else ""
        L.append(f"{head}{spk_name(p['spk'])}{colon}{p['text']}")
        L.append("")
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(L))


def write_srt(path, paras, english=False):
    """字幕：每一句用辨識器給的真實時間；太長的句子依標點再切，時間按字數分配。"""
    L = []
    n = 0
    colon = ": " if english else "："
    limit = 84 if english else 30
    for p in paras:
        for a, b, text in p["cues"]:
            text = text.strip() if english else tidy(text)
            sents = re.findall(r"[^。？！?!，,]+[。？！?!，,]*", text) or [text]
            lines, cur = [], ""
            for x in sents:
                if cur and len(cur) + len(x) > limit:
                    lines.append(cur)
                    cur = ""
                cur += x
            if cur:
                lines.append(cur)
            tot = sum(len(x) for x in lines) or 1
            acc = 0
            for x in lines:
                ta = a + (b - a) * acc / tot
                acc += len(x)
                tb = a + (b - a) * acc / tot
                n += 1
                L += [str(n), f"{fmt_ts(ta, True)} --> {fmt_ts(tb, True)}",
                      f"{spk_name(p['spk'])}{colon}{x.strip()}", ""]
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------------ 主流程
class Engine:
    """模型只載入一次，可以連續處理多個檔案。"""

    def __init__(self, models_dir=None, threads=None, device="auto", log=print, ui="zh"):
        self.msg = MSG.get(ui, MSG["zh"])
        self.m = find_models(models_dir)
        self.threads = threads or default_threads()
        self.device = device
        self.log = log or (lambda *_: None)
        self._rec = self._dia = self._pun = None

    @property
    def rec(self):
        if self._rec is None:
            self._rec = Recognizer(self.m["asr"], self.threads, self.device, self.log, self.msg)
        return self._rec

    @property
    def dia(self):
        if self._dia is None:
            self._dia = Diarizer(self.m["seg"], self.m["spk"], self.threads)
        return self._dia

    @property
    def pun(self):
        if self._pun is None:
            self._pun = Punctuator(self.m["punct"], self.threads)
        return self._pun

    def transcribe(self, src, out_base, language="zh", num_speakers=0, hotwords="",
                   want_txt=True, want_srt=True, timestamps=True,
                   progress=None, cancel=None):
        """progress(fraction, stage_text)。回傳輸出檔案列表。"""
        prog = progress or (lambda *_: None)
        W = {"dec": (0.0, 0.03), "dia": (0.03, 0.20), "asr": (0.20, 0.97), "fmt": (0.97, 1.0)}

        M = self.msg

        def stage(name):
            a, b = W[name]
            return lambda f: prog(a + (b - a) * f, M[name])

        t0 = time.time()
        stage("dec")(0)
        audio = load_audio(src)
        dur = len(audio) / SR
        if dur < 0.5:
            raise ValueError(M["too_short"])
        self.log(M["length"].format(t=fmt_ts(dur)))

        if num_speakers == 1:
            turns = [(0.0, dur, 0)]
        else:
            stage("dia")(0)
            turns = self.dia.run(audio, num_speakers, stage("dia"), cancel)
        nspk = len({t[2] for t in turns}) or 1
        self.log(M["speakers"].format(n=nspk, s=time.time() - t0))

        stage("asr")(0)
        _ = self.rec
        self.log(M["device"].format(d="GPU" if self.rec.device == "cuda" else "CPU"))
        segs, lang = self.rec.run(audio, language, hotwords, stage("asr"), cancel)
        if not segs:
            raise ValueError(M["no_speech"])
        english = (lang == "en")

        stage("fmt")(0)
        paras, nspk = compose(segs, turns, lang, self.pun)

        outs = []
        if want_txt:
            p = out_base + ".txt"
            write_txt(p, paras, src, dur, nspk, timestamps, english)
            outs.append(p)
        if want_srt:
            p = out_base + ".srt"
            write_srt(p, paras, english)
            outs.append(p)
        prog(1.0, M["done"])
        self.log(M["finished"].format(n=len(paras), s=time.time() - t0))
        return outs


_CC = None
# 這些字在繁體裡本來就是正確的，OpenCC 會把「台」轉成「臺」、「里」轉成「裡」等，保留原字
_KEEP = set("台里干后面系才只周游松准云采")


def to_traditional(text):
    """模型偶爾會吐出簡體字，轉成台灣繁體；原本就是繁體的字不動。"""
    global _CC
    try:
        if _CC is None:
            from opencc import OpenCC
            _CC = OpenCC("s2tw")
        t = _CC.convert(text)
        if len(t) != len(text):
            return t
        return "".join(o if o in _KEEP else c for o, c in zip(text, t))
    except Exception:
        return text
