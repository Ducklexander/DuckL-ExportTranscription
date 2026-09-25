# DuckL-ExportTranscription

A small Windows tool that turns recordings and videos into a clean transcript,
with punctuation, paragraphs and speaker labels (A, B, C...).
Tuned for Taiwanese Mandarin, also handles English and mixed Chinese/English.
Runs fully offline. No install, no Python, no account, nothing is uploaded.
The interface switches between Chinese and English in the same exe.

## Download

From the [latest release](../../releases/latest):

| File | What it is |
| --- | --- |
| `DuckL-ExportTranscription.zip` | The tool, models included. Unzip anywhere and run the exe |
| `DuckL-ExportTranscription_GPU.zip` | Optional, NVIDIA cuBLAS. You normally do not need to download this by hand |

That is the whole setup: unzip, run. On a PC with an NVIDIA card the window shows
an "Install GPU acceleration" button. It downloads NVIDIA cuBLAS (from NVIDIA's
own server, falling back to the GPU zip in this release), puts it in place and
restarts, for about 5x speed. It took under a minute here. For an offline PC, download the
GPU zip yourself and unzip it to the same place as the main zip.

The build is unsigned, so SmartScreen may warn on first run: choose "More info"
then "Run anyway".

## What you get

```
[00:01:23] A：可以問一下通道打開大概是怎麼樣的嗎？

[00:01:29] B：好，讓我找到那個，好，就這個，然後蘭陵王就是演完，我們就讓他退場，
就是看要怎麼退都可以的，反正就是，他就消失好了，或粒子效果消失什麼都可以。
```

A `.txt` transcript and a `.srt` subtitle file, saved next to the source with a
`_逐字稿` suffix. Nothing is ever overwritten.

## How it works

| Step | Model | Runtime |
| --- | --- | --- |
| Decode wav / mp3 / mp4 / m4a / mov ... | | PyAV (no ffmpeg install needed) |
| Speech recognition | [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25), Whisper fine-tuned for Taiwanese Mandarin and code-switching, int8 | CTranslate2 via faster-whisper |
| Speaker diarization | pyannote segmentation 3.0 + 3D-Speaker CAM++ | sherpa-onnx |
| Punctuation | FunASR CT-Transformer zh-en, int8 | sherpa-onnx |

Diarization runs twice: first deliberately over-split, then clusters are merged
again by voice-print centroid, which picks the number of speakers on its own.
Speakers are assigned per recognised sentence, so a speaker change never lands
in the middle of a phrase. Output is normalised to Traditional Chinese (Taiwan).

1.5 GB zipped, 1.8 GB unzipped. Measured on a 21:41 two-person meeting:
about 25 minutes on a 6-core laptop CPU, 4.5 minutes with the GPU pack on an RTX 4060.
The GPU pack is only cuBLAS; CTranslate2's Whisper does not need cuDNN.

## Usage

Drag files in, optionally pick the number of speakers and add names or terms
that should be spelled correctly, then start. Settings and the interface
language are remembered in `%APPDATA%\DuckL-ExportTranscription`.

Key terms matter more than anything else for names: in the sample meeting,
adding 蘭陵王 turned 藍寧光 and 蘭陵娃 into the right name everywhere.

Command line, for batch use:

```
DuckL-ExportTranscription.exe --cli talk.mp4 --speakers 2 --hotwords "蘭陵王,Unity"
DuckL-ExportTranscription.exe --cli talk.mp4 --lang en --ui en
DuckL-ExportTranscription.exe --install-gpu      (GPU acceleration without the window)
```

There is no console, so messages go to a `.log` next to the output.

## Build

```
pip install faster-whisper sherpa-onnx opencc-python-reimplemented tkinterdnd2 nuitka pillow
python build.py          # compile and zip the tool
python build.py --gpu    # zip the GPU pack from _work/gpu/nvidia/cublas*.dll
```

Requires MSVC. Models go in `models/` (see `models/模型來源.txt` in the release);
they are not in this repository because of their size.

## License

Free to use. Models keep their own licenses (Apache-2.0 / MIT).

---

Developed with [Claude](https://claude.ai/code).
