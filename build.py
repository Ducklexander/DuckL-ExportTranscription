# -*- coding: utf-8 -*-
"""用 Nuitka 把 ExportTranscription 編譯成原生碼資料夾版，並打包成 zip。

  DuckL-ExportTranscription.zip
    DuckL-ExportTranscription/
      DuckL-ExportTranscription.exe
      models/            語音辨識、說話人、標點模型（約 1.6 GB）
      使用說明.txt / Readme_EN.txt
      (執行所需的 DLL)

用法： python build.py          （需要 Visual Studio / MSVC）
       python build.py --zip-only   只重新打 zip
       python build.py --gpu        另外打一包 GPU 加速包（需要 _work/gpu 裡的 NVIDIA DLL）
"""
import os, sys, shutil, subprocess, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_nuitka")
NAME = "DuckL-ExportTranscription"
VSTR = "1.1.0.0"
DIST = os.path.join(OUT, "exporttranscription.dist")


def make_icon(path):
    """畫一個簡單的圖示：對話泡泡 + 聲波。"""
    from PIL import Image, ImageDraw
    S = 256
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    blue = (40, 110, 220, 255)
    d.rounded_rectangle([20, 30, 236, 190], radius=36, fill=blue)
    d.polygon([(64, 180), (64, 236), (120, 184)], fill=blue)
    w = (255, 255, 255, 255)
    for x, h in ((68, 34), (98, 70), (128, 100), (158, 60), (188, 40)):
        d.rounded_rectangle([x - 9, 110 - h // 2, x + 9, 110 + h // 2], radius=9, fill=w)
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    im.save(path, sizes=sizes)
    return path


def compile_exe():
    os.makedirs(OUT, exist_ok=True)
    ico = make_icon(os.path.join(OUT, "icon.ico"))
    args = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--msvc=latest",
        "--assume-yes-for-downloads",
        "--windows-console-mode=disable",
        "--enable-plugin=tk-inter",
        "--include-package=tkinterdnd2",
        "--include-module=engine",
        "--include-package=faster_whisper",
        "--include-package-data=faster_whisper",
        "--include-package=sherpa_onnx",
        "--include-package-data=opencc",
        "--include-package=opencc",
        f"--windows-icon-from-ico={ico}",
        "--company-name=DuckL",
        "--product-name=ExportTranscription",
        f"--file-version={VSTR}",
        f"--product-version={VSTR}",
        "--file-description=ExportTranscription - local speech to transcript",
        "--copyright=Free to use",
        f"--output-dir={OUT}",
        f"--output-filename={NAME}.exe",
    ]
    for m in ("torch", "matplotlib", "scipy", "pandas", "PyQt5", "PyQt6", "PySide2",
              "PySide6", "IPython", "pytest", "sqlalchemy", "rich", "bs4", "setuptools",
              "pip", "unittest", "doctest", "pydoc", "onnx", "nuitka", "sympy",
              "transformers", "PIL", "hf_xet"):
        args.append(f"--nofollow-import-to={m}")
    args.append(os.path.join(HERE, "exporttranscription.py"))
    print(" ".join(args), flush=True)
    rc = subprocess.call(args)
    print("nuitka rc =", rc, flush=True)
    return rc


def copy_payload():
    # 模型：只放執行需要的檔案
    src = os.path.join(HERE, "models")
    dst = os.path.join(DIST, "models")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for sub in ("asr", "diarization", "punct"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    for fn in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json",
               "preprocessor_config.json"):
        shutil.copy2(os.path.join(src, "asr", fn), os.path.join(dst, "asr", fn))
    for fn in os.listdir(os.path.join(src, "diarization")):
        shutil.copy2(os.path.join(src, "diarization", fn), os.path.join(dst, "diarization", fn))
    shutil.copy2(os.path.join(src, "punct", "model.int8.onnx"),
                 os.path.join(dst, "punct", "model.int8.onnx"))
    with open(os.path.join(dst, "模型來源.txt"), "w", encoding="utf-8") as f:
        f.write(MODEL_NOTICE)
    for fn in ("使用說明.txt", "Readme_EN.txt"):
        if os.path.exists(os.path.join(HERE, fn)):
            shutil.copy2(os.path.join(HERE, fn), DIST)


MODEL_NOTICE = """模型來源 / Model sources
asr/          MediaTek-Research/Breeze-ASR-25（Apache-2.0），CTranslate2 int8 轉換
diarization/  segmentation.onnx: pyannote/segmentation-3.0（MIT，sherpa-onnx 轉換）
              speaker.onnx: 3D-Speaker CAM++ zh_en common advanced（Apache-2.0）
punct/        FunASR CT-Transformer zh-en（sherpa-onnx 轉換，int8 量化）
"""


def make_zip(src_dir, zpath, arc_root):
    if os.path.exists(zpath):
        os.remove(zpath)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as z:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.join(arc_root, os.path.relpath(p, src_dir)))
                n += 1
    print(f"==> {zpath} {os.path.getsize(zpath) / 1048576:.0f} MB ({n} files)", flush=True)


def make_gpu_pack():
    """NVIDIA 顯卡加速包：cuBLAS 的 DLL，解壓到主程式資料夾即可。"""
    src = os.path.join(HERE, "_work", "gpu", "nvidia")
    # 實測 CTranslate2 的 Whisper 在 GPU 上只需要 cuBLAS，不需要 cuDNN
    want = ("cublas64_12.dll", "cublasLt64_12.dll")
    stage = os.path.join(OUT, "gpu_pack")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    found = 0
    for root, _, files in os.walk(src):
        for fn in files:
            if fn in want:
                shutil.copy2(os.path.join(root, fn), os.path.join(stage, fn))
                found += 1
    print("gpu dlls:", found, flush=True)
    make_zip(stage, os.path.join(HERE, f"{NAME}_GPU.zip"), NAME)


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        make_gpu_pack()
        sys.exit(0)
    if "--zip-only" not in sys.argv:
        rc = compile_exe()
        if rc != 0:
            sys.exit(rc)
    copy_payload()
    make_zip(DIST, os.path.join(HERE, f"{NAME}.zip"), NAME)
