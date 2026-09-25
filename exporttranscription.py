# -*- coding: utf-8 -*-
"""
ExportTranscription - 本機、免費的 AI 語音轉逐字稿
把錄音 / 影片拖進來，輸出有標點、分段、標好說話人（A、B、C…）的逐字稿。
全部在本機執行，不上傳、不需要網路。介面可切換中文 / English。
"""
import os, sys, json, time, queue, threading, traceback, subprocess

APP = "ExportTranscription"
VER = "1.1"

MEDIA_EXT = (".wav", ".mp3", ".mp4", ".m4a", ".aac", ".flac", ".ogg", ".opus",
             ".wma", ".mov", ".mkv", ".webm", ".avi", ".3gp", ".amr")

LANG_CODES = ["zh", "en", "auto"]

STRINGS = {
    "zh": {
        "title": "語音轉逐字稿",
        "drop_hint": "  把錄音或影片拖進來（wav / mp3 / mp4 / m4a …）",
        "pick_hint": "  按「加入檔案」選擇錄音或影片",
        "add": "加入檔案", "remove": "移除選取", "clear": "清空",
        "choose": "選擇錄音或影片", "media": "錄音 / 影片", "all": "所有檔案",
        "settings": "設定", "language": "語言",
        "languages": ["中文（台灣）", "English", "自動偵測"],
        "speakers": "說話人數",
        "speaker_list": ["自動", "1 人", "2 人", "3 人", "4 人", "5 人", "6 人", "7 人", "8 人"],
        "speaker_note": "知道人數就選，分得更準",
        "hotwords": "專有名詞",
        "hot_note": "（選填）人名、作品名、術語，用逗號隔開，例如：蘭陵王, Unity, 動線",
        "output": "輸出", "txt": "逐字稿 .txt", "srt": "字幕 .srt",
        "ts": "段落加時間", "open": "完成後打開",
        "outdir": "輸出資料夾", "outdir_note": "（留空 = 存在原檔旁邊，檔名加「_逐字稿」）",
        "start": "開始轉逐字稿", "stop": "停止", "ready": "就緒", "stopping": "停止中…",
        "loading": "載入模型…", "load_fail": "✖ 無法載入模型：{e}", "failed_status": "失敗",
        "cancelled": "  ✖ 已取消", "failed": "  ✖ 失敗：{e}",
        "summary": "完成 {ok}/{n}（{t}）", "all_done": "—— 全部完成 ——\n",
        "eta": "，約剩 {t}",
        "need_files": "請先加入錄音或影片檔", "need_format": "請至少勾選一種輸出格式",
        "no_outdir": "輸出資料夾不存在",
        "banner": "{app} {ver}｜全部在這台電腦上執行，不上傳、不需要網路。",
        "banner2": "第一次處理會先載入模型（約 10 秒）。一般筆電 CPU 大約是錄音長度的 1～1.5 倍時間。",
        "no_dnd": "（此環境無法拖放，請用「加入檔案」）",
        "suffix": "_逐字稿",
        "gpu_checking": "", "gpu_none": "處理器：CPU",
        "gpu_ok": "GPU 加速：已啟用（NVIDIA）",
        "gpu_missing": "偵測到 NVIDIA 顯卡，安裝 GPU 加速可快約 5 倍",
        "gpu_install": "安裝 GPU 加速（約 540 MB）",
        "gpu_confirm": "要下載並安裝 GPU 加速嗎？\n\n"
                       "會從 NVIDIA 官方網站下載 cuBLAS（約 540 MB），"
                       "放進程式資料夾。完成後程式會自動重新開啟。",
        "gpu_dl": "下載 GPU 加速 {a} / {b} MB",
        "gpu_done": "GPU 加速安裝完成，程式將重新開啟。",
        "gpu_fail": "GPU 加速安裝失敗：{e}\n\n"
                    "也可以手動從 GitHub Release 下載 DuckL-ExportTranscription_GPU.zip，"
                    "解壓到程式資料夾。",
        "busy": "目前正在處理中，請等完成或先停止",
    },
    "en": {
        "title": "Speech to Transcript",
        "drop_hint": "  Drop recordings or videos here (wav / mp3 / mp4 / m4a ...)",
        "pick_hint": "  Click \"Add files\" to choose recordings or videos",
        "add": "Add files", "remove": "Remove", "clear": "Clear",
        "choose": "Choose recordings or videos", "media": "Audio / video", "all": "All files",
        "settings": "Settings", "language": "Language",
        "languages": ["Chinese (Taiwan)", "English", "Auto detect"],
        "speakers": "Speakers",
        "speaker_list": ["Auto", "1", "2", "3", "4", "5", "6", "7", "8"],
        "speaker_note": "pick it if you know, it helps",
        "hotwords": "Key terms",
        "hot_note": "(optional) names and terms to spell right, comma separated, e.g. Unity, Kubernetes",
        "output": "Output", "txt": "Transcript .txt", "srt": "Subtitles .srt",
        "ts": "Timestamps", "open": "Open when done",
        "outdir": "Output folder", "outdir_note": "(empty = next to the source, name gets _transcript)",
        "start": "Transcribe", "stop": "Stop", "ready": "Ready", "stopping": "Stopping...",
        "loading": "Loading models...", "load_fail": "x Could not load models: {e}",
        "failed_status": "Failed",
        "cancelled": "  x Cancelled", "failed": "  x Failed: {e}",
        "summary": "Finished {ok}/{n} ({t})", "all_done": "-- All done --\n",
        "eta": ", about {t} left",
        "need_files": "Add some recordings or videos first",
        "need_format": "Tick at least one output format",
        "no_outdir": "Output folder does not exist",
        "banner": "{app} {ver} | Runs entirely on this computer. Nothing is uploaded.",
        "banner2": "The first file loads the models (about 10 s). On a laptop CPU it takes "
                   "roughly 1 to 1.5 times the recording length.",
        "no_dnd": "(drag and drop unavailable here, use \"Add files\")",
        "suffix": "_transcript",
        "gpu_checking": "", "gpu_none": "Running on: CPU",
        "gpu_ok": "GPU acceleration: on (NVIDIA)",
        "gpu_missing": "NVIDIA GPU found. GPU acceleration makes it about 5x faster",
        "gpu_install": "Install GPU acceleration (about 540 MB)",
        "gpu_confirm": "Download and install GPU acceleration?\n\n"
                       "This downloads NVIDIA cuBLAS (about 540 MB) from NVIDIA into the "
                       "program folder. The program restarts when it is done.",
        "gpu_dl": "Downloading GPU acceleration {a} / {b} MB",
        "gpu_done": "GPU acceleration installed. The program will restart.",
        "gpu_fail": "Could not install GPU acceleration: {e}\n\n"
                    "You can also download DuckL-ExportTranscription_GPU.zip from the "
                    "GitHub release and unzip it into the program folder.",
        "busy": "Busy. Wait until it finishes or stop it first",
    },
}


# ------------------------------------------------------------------ 設定
def settings_path():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "DuckL-ExportTranscription", "settings.json")


def load_settings():
    try:
        with open(settings_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def system_ui_lang():
    """Windows 顯示語言是中文就用中文，否則英文。"""
    try:
        import ctypes
        lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        return "zh" if (lid & 0x3FF) == 0x04 else "en"
    except Exception:
        return "zh"


def human_time(s):
    s = int(s)
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def out_base_for(src, odir, suffix="_逐字稿"):
    base = os.path.splitext(os.path.basename(src))[0] + suffix
    d = odir or os.path.dirname(src)
    p = os.path.join(d, base)
    k = 2
    while os.path.exists(p + ".txt") or os.path.exists(p + ".srt"):
        p = os.path.join(d, f"{base}({k})")
        k += 1
    return p


def restart_self():
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        cmd = [os.path.abspath(sys.argv[0])]
    else:
        cmd = [sys.executable, os.path.abspath(__file__)]
    subprocess.Popen(cmd, close_fds=True)


# ------------------------------------------------------------------ 視窗
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    cfg = load_settings()
    ui = {"lang": cfg.get("ui") if cfg.get("ui") in STRINGS else system_ui_lang()}

    def T(k):
        return STRINGS[ui["lang"]][k]

    DND = True
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
    except Exception:
        DND = False
        root = tk.Tk()

    root.geometry("780x640")
    root.minsize(700, 580)

    files = []
    worker = {"th": None, "cancel": None, "engine": None, "gpu": "checking"}
    msgq = queue.Queue()
    texts = []          # (widget, key, 選項)：切換語言時重設文字

    def txt(widget, key, **opt):
        texts.append((widget, key))
        widget.config(text=T(key), **opt)
        return widget

    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass

    # 最上面：語言切換
    bar_top = ttk.Frame(root, padding=(10, 8, 10, 0))
    bar_top.pack(fill="x")
    ui_var = tk.StringVar(value=ui["lang"])
    ttk.Radiobutton(bar_top, text="English", value="en", variable=ui_var,
                    command=lambda: set_lang("en")).pack(side="right")
    ttk.Radiobutton(bar_top, text="中文", value="zh", variable=ui_var,
                    command=lambda: set_lang("zh")).pack(side="right", padx=(0, 8))

    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    listwrap = ttk.Frame(top)
    listwrap.pack(side="left", fill="both", expand=True)
    lb = tk.Listbox(listwrap, height=6, activestyle="none", selectmode="extended")
    lb.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(listwrap, command=lb.yview)
    sb.pack(side="left", fill="y")
    lb.config(yscrollcommand=sb.set)

    btns = ttk.Frame(top)
    btns.pack(side="left", fill="y", padx=(8, 0))

    def refresh():
        lb.delete(0, "end")
        if not files:
            lb.insert("end", T("drop_hint") if DND else T("pick_hint"))
        for f in files:
            try:
                sz = f"{os.path.getsize(f) / 1048576:.1f} MB"
            except Exception:
                sz = "?"
            lb.insert("end", f"{os.path.basename(f)}   [{sz}]")

    def add_files(paths):
        for p in paths:
            try:
                p = os.path.abspath(p)
            except Exception:
                continue
            if os.path.isdir(p):
                add_files(os.path.join(p, f) for f in sorted(os.listdir(p)))
                continue
            if p.lower().endswith(MEDIA_EXT) and os.path.isfile(p) and p not in files:
                files.append(p)
        refresh()

    def pick():
        pat = " ".join("*" + e for e in MEDIA_EXT)
        add_files(filedialog.askopenfilenames(
            title=T("choose"), filetypes=[(T("media"), pat), (T("all"), "*.*")]))

    def remove_sel():
        for i in sorted(lb.curselection(), reverse=True):
            if i < len(files):
                del files[i]
        refresh()

    txt(ttk.Button(btns, command=pick, width=12), "add").pack(pady=2)
    txt(ttk.Button(btns, command=remove_sel, width=12), "remove").pack(pady=2)
    txt(ttk.Button(btns, width=12, command=lambda: (files.clear(), refresh())),
        "clear").pack(pady=2)

    if DND:
        def on_drop(ev):
            add_files(root.tk.splitlist(ev.data))
        for w in (lb, root):
            try:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", on_drop)
            except Exception:
                pass

    opt = ttk.LabelFrame(root, padding=10)
    texts.append((opt, "settings"))
    opt.pack(fill="x", padx=10)

    txt(ttk.Label(opt), "language").grid(row=0, column=0, sticky="w")
    lang_cb = ttk.Combobox(opt, state="readonly", width=16)
    lang_cb.grid(row=0, column=1, sticky="w", padx=6)

    txt(ttk.Label(opt), "speakers").grid(row=0, column=2, sticky="e", padx=(16, 0))
    spk_cb = ttk.Combobox(opt, state="readonly", width=8)
    spk_cb.grid(row=0, column=3, sticky="w", padx=6)
    txt(ttk.Label(opt), "speaker_note").grid(row=0, column=4, sticky="w")

    hot = tk.StringVar(value=cfg.get("hotwords", ""))
    txt(ttk.Label(opt), "hotwords").grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(opt, textvariable=hot).grid(row=1, column=1, columnspan=4, sticky="we",
                                          padx=6, pady=(8, 0))
    txt(ttk.Label(opt, foreground="#666"), "hot_note").grid(
        row=2, column=1, columnspan=4, sticky="w", padx=6)

    want_txt = tk.BooleanVar(value=cfg.get("txt", True))
    want_srt = tk.BooleanVar(value=cfg.get("srt", True))
    want_ts = tk.BooleanVar(value=cfg.get("ts", True))
    want_open = tk.BooleanVar(value=cfg.get("open", True))
    outs = ttk.Frame(opt)
    outs.grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))
    txt(ttk.Label(outs), "output").pack(side="left")
    txt(ttk.Checkbutton(outs, variable=want_txt), "txt").pack(side="left", padx=(12, 0))
    txt(ttk.Checkbutton(outs, variable=want_srt), "srt").pack(side="left", padx=(8, 0))
    txt(ttk.Checkbutton(outs, variable=want_ts), "ts").pack(side="left", padx=(8, 0))
    txt(ttk.Checkbutton(outs, variable=want_open), "open").pack(side="left", padx=(8, 0))

    outdir = tk.StringVar(value="")
    txt(ttk.Label(opt), "outdir").grid(row=4, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(opt, textvariable=outdir).grid(row=4, column=1, columnspan=3,
                                             sticky="we", padx=6, pady=(8, 0))
    ttk.Button(opt, text="…", width=3,
               command=lambda: outdir.set(filedialog.askdirectory() or outdir.get())
               ).grid(row=4, column=4, sticky="w", pady=(8, 0))
    txt(ttk.Label(opt, foreground="#666"), "outdir_note").grid(
        row=5, column=1, columnspan=4, sticky="w", padx=6)
    opt.columnconfigure(4, weight=1)

    # GPU 狀態列
    gpu_row = ttk.Frame(root, padding=(10, 6, 10, 0))
    gpu_row.pack(fill="x")
    gpu_lbl = ttk.Label(gpu_row, text="")
    gpu_lbl.pack(side="left")
    gpu_btn = ttk.Button(gpu_row)
    texts.append((gpu_btn, "gpu_install"))

    bar = ttk.Progressbar(root, mode="determinate", maximum=1000)
    bar.pack(fill="x", padx=10, pady=(8, 4))

    act = ttk.Frame(root)
    act.pack(fill="x", padx=10)
    run_btn = txt(ttk.Button(act, width=18), "start")
    run_btn.pack(side="left")
    stop_btn = txt(ttk.Button(act, width=10, state="disabled"), "stop")
    stop_btn.pack(side="left", padx=6)
    status = ttk.Label(act, text="")
    status.pack(side="left", padx=10)

    logbox = tk.Text(root, height=9, wrap="word", state="disabled",
                     font=("Microsoft JhengHei UI", 9))
    logbox.pack(fill="both", expand=True, padx=10, pady=10)

    def show_gpu():
        st = worker["gpu"]
        gpu_btn.pack_forget()
        if st == "ok":
            gpu_lbl.config(text=T("gpu_ok"))
        elif st == "missing":
            gpu_lbl.config(text=T("gpu_missing"))
            gpu_btn.pack(side="left", padx=10)
        elif st == "none":
            gpu_lbl.config(text=T("gpu_none"))
        else:
            gpu_lbl.config(text="")

    def apply_texts():
        root.title(f"{APP} {VER} - {T('title')}")
        for w, k in texts:
            w.config(text=T(k))
        li, si = lang_cb.current(), spk_cb.current()
        lang_cb.config(values=T("languages"))
        spk_cb.config(values=T("speaker_list"))
        lang_cb.current(max(0, li))
        spk_cb.current(max(0, si))
        if worker["th"] is None:
            status.config(text=T("ready"))
        refresh()
        show_gpu()

    def set_lang(code):
        if code == ui["lang"]:
            return
        ui["lang"] = code
        apply_texts()
        if not ui.get("used"):              # 還沒開始處理：開頭說明也換語言
            logbox.config(state="normal")
            logbox.delete("1.0", "end")
            logbox.config(state="disabled")
            banner()
        persist()

    def banner():
        log(T("banner").format(app=APP, ver=VER))
        log(T("banner2"))
        if not DND:
            log(T("no_dnd"))

    def persist():
        save_settings({
            "ui": ui["lang"], "lang": LANG_CODES[max(0, lang_cb.current())],
            "speakers": max(0, spk_cb.current()), "hotwords": hot.get().strip(),
            "txt": want_txt.get(), "srt": want_srt.get(), "ts": want_ts.get(),
            "open": want_open.get()})

    def log(msg):
        msgq.put(("log", msg))

    def pump():
        try:
            while True:
                kind, val = msgq.get_nowait()
                if kind == "log":
                    logbox.config(state="normal")
                    logbox.insert("end", str(val) + "\n")
                    logbox.see("end")
                    logbox.config(state="disabled")
                elif kind == "prog":
                    bar["value"] = max(0, min(1000, val * 1000))
                elif kind == "status":
                    status.config(text=val)
                elif kind == "open":
                    try:
                        os.startfile(val)
                    except Exception:
                        pass
                elif kind == "gpu":
                    worker["gpu"] = val
                    show_gpu()
                elif kind == "gpu_done":
                    messagebox.showinfo(APP, T("gpu_done"))
                    persist()
                    restart_self()
                    root.destroy()
                    return
                elif kind == "gpu_fail":
                    if val is not None:         # None = 使用者按了停止
                        messagebox.showerror(APP, T("gpu_fail").format(e=val))
                    gpu_btn.config(state="normal")
                    run_btn.config(state="normal")
                    stop_btn.config(state="disabled")
                    bar["value"] = 0
                    worker["th"] = None
                    status.config(text=T("ready"))
                elif kind == "done":
                    run_btn.config(state="normal")
                    stop_btn.config(state="disabled")
                    worker["th"] = None
        except queue.Empty:
            pass
        root.after(100, pump)

    # ---------------------------------------------------------------- 轉錄
    def job(items, opts, odir, open_after, lang_ui):
        import engine
        S = STRINGS[lang_ui]
        cancel = worker["cancel"]
        n = len(items)
        ok = 0
        t0 = time.time()
        first_out = None
        try:
            if worker["engine"] is None or worker["engine"].msg is not engine.MSG[lang_ui]:
                msgq.put(("status", S["loading"]))
                old = worker["engine"]
                worker["engine"] = engine.Engine(log=log, ui=lang_ui)
                if old is not None:          # 換語言時沿用已載入的模型
                    worker["engine"]._rec, worker["engine"]._dia, worker["engine"]._pun = \
                        old._rec, old._dia, old._pun
            eng = worker["engine"]
        except Exception as e:
            log(S["load_fail"].format(e=e))
            msgq.put(("status", S["failed_status"]))
            msgq.put(("done", None))
            return
        for idx, src in enumerate(items):
            if cancel.is_set():
                break
            log(f"[{idx + 1}/{n}] {os.path.basename(src)}")
            ts = time.time()

            def prog(f, text, i=idx, ts=ts):
                msgq.put(("prog", (i + f) / n))
                el = time.time() - ts
                eta = S["eta"].format(t=human_time(el * (1 - f) / f)) if f > 0.25 else ""
                msgq.put(("status", f"({i + 1}/{n}) {text} {f * 100:.0f}%{eta}"))

            base = out_base_for(src, odir, S["suffix"])
            try:
                res = eng.transcribe(src, base, progress=prog, cancel=cancel, **opts)
                for p in res:
                    log(f"  → {p}")
                if res and first_out is None:
                    first_out = res[0]
                ok += 1
            except engine.Cancelled:
                log(S["cancelled"])
                break
            except Exception as e:
                log(S["failed"].format(e=e))
                log(traceback.format_exc(limit=3))
        msgq.put(("prog", 1.0))
        msgq.put(("status", S["summary"].format(ok=ok, n=n, t=human_time(time.time() - t0))))
        log(S["all_done"])
        if open_after and first_out and not cancel.is_set():
            msgq.put(("open", first_out))
        msgq.put(("done", None))

    def start():
        if worker["th"]:
            return
        if not files:
            messagebox.showinfo(APP, T("need_files"))
            return
        if not (want_txt.get() or want_srt.get()):
            messagebox.showinfo(APP, T("need_format"))
            return
        odir = outdir.get().strip()
        if odir and not os.path.isdir(odir):
            messagebox.showerror(APP, T("no_outdir"))
            return
        persist()
        ui["used"] = True
        opts = dict(language=LANG_CODES[lang_cb.current()],
                    num_speakers=spk_cb.current(),
                    hotwords=hot.get().strip(),
                    want_txt=want_txt.get(), want_srt=want_srt.get(),
                    timestamps=want_ts.get())
        worker["cancel"] = threading.Event()
        run_btn.config(state="disabled")
        stop_btn.config(state="normal")
        bar["value"] = 0
        th = threading.Thread(target=job, daemon=True,
                              args=(list(files), opts, odir, want_open.get(), ui["lang"]))
        worker["th"] = th
        th.start()

    def stop():
        if worker["cancel"]:
            worker["cancel"].set()
            status.config(text=T("stopping"))

    # ---------------------------------------------------------------- GPU
    def check_gpu():
        try:
            import engine
            msgq.put(("gpu", engine.gpu_state()))
        except Exception:
            msgq.put(("gpu", "none"))

    def install_gpu():
        if worker["th"]:
            messagebox.showinfo(APP, T("busy"))
            return
        if not messagebox.askokcancel(APP, T("gpu_confirm")):
            return
        gpu_btn.config(state="disabled")
        run_btn.config(state="disabled")
        stop_btn.config(state="normal")
        worker["cancel"] = threading.Event()
        S = STRINGS[ui["lang"]]

        def work():
            import engine
            try:
                def prog(a, b):
                    if b:
                        msgq.put(("prog", a / b))
                    msgq.put(("status", S["gpu_dl"].format(a=a >> 20, b=(b >> 20) or "?")))
                engine.install_gpu(prog, worker["cancel"])
                msgq.put(("gpu_done", None))
            except engine.Cancelled:
                msgq.put(("gpu_fail", None))
            except Exception as e:
                msgq.put(("gpu_fail", e))

        th = threading.Thread(target=work, daemon=True)
        worker["th"] = th
        th.start()

    run_btn.config(command=start)
    stop_btn.config(command=stop)
    gpu_btn.config(command=install_gpu)

    lang_cb.config(values=T("languages"))
    spk_cb.config(values=T("speaker_list"))
    lang_cb.current(LANG_CODES.index(cfg["lang"]) if cfg.get("lang") in LANG_CODES
                    else (1 if ui["lang"] == "en" else 0))
    spk_cb.current(min(8, max(0, int(cfg.get("speakers", 0)))))
    apply_texts()
    banner()
    add_files(sys.argv[1:])
    threading.Thread(target=check_gpu, daemon=True).start()

    def on_close():
        persist()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    pump()
    root.mainloop()


# ------------------------------------------------------------------ 命令列
def cli(argv):
    """ExportTranscription.exe --cli 檔案 [-o 輸出檔名(不含副檔名)] [--speakers N]
       [--lang zh|en|auto] [--hotwords "詞1,詞2"] [--ui zh|en] [--no-srt] [--no-txt]
    視窗程式沒有 console，訊息寫到輸出旁的 .log。"""
    import engine
    src = dst = None
    ui = "zh"
    opts = dict(language="zh", num_speakers=0, hotwords="")
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-o":
            i += 1
            dst = argv[i]
        elif a == "--speakers":
            i += 1
            opts["num_speakers"] = int(argv[i])
        elif a == "--lang":
            i += 1
            opts["language"] = argv[i]
        elif a == "--ui":
            i += 1
            ui = "en" if argv[i].lower().startswith("en") else "zh"
        elif a == "--hotwords":
            i += 1
            opts["hotwords"] = argv[i]
        elif a == "--no-srt":
            opts["want_srt"] = False
        elif a == "--no-txt":
            opts["want_txt"] = False
        elif not a.startswith("-"):
            src = a
        i += 1
    if not src:
        return 2
    dst = dst or out_base_for(src, "", STRINGS[ui]["suffix"])
    lines = []
    t0 = time.time()

    def log(m):
        lines.append(str(m))
    try:
        eng = engine.Engine(log=log, ui=ui)
        eng.transcribe(src, dst, **opts)
        log(f"OK {time.time() - t0:.0f}s")
        rc = 0
    except Exception:
        log(traceback.format_exc())
        rc = 1
    try:
        with open(dst + ".log", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    if "--cli" in sys.argv:
        a = list(sys.argv[1:])
        a.remove("--cli")
        sys.exit(cli(a))
    main()
