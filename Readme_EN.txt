ExportTranscription 1.1
=======================

Free, offline speech to transcript. Unzip and double-click
DuckL-ExportTranscription.exe. No install, no Python, nothing is uploaded.

1. Switch the interface to English with the radio button at the top right
   (it is remembered).
2. Drop recordings or videos into the list (wav, mp3, mp4, m4a, mov, flac ...).
3. Pick the language (Chinese (Taiwan) / English / Auto), the number of speakers
   if you know it, and optionally key terms that should be spelled right.
4. Click "Transcribe". The transcript opens when it is done.

Output is saved next to the source, never overwriting anything:
  name_transcript.txt   paragraphs with time and speaker, e.g.
                        [00:01:23] A: Did you change the player path again?
  name_transcript.srt   subtitles

Speed: on a laptop CPU about 1 to 1.5 times the recording length. With an
NVIDIA GPU, click "Install GPU acceleration" in the window (downloads about
540 MB, then restarts) for about 5x speed. Offline machines can download
DuckL-ExportTranscription_GPU.zip from the GitHub release and unzip it to the
same place.

Keep the models folder next to the exe. Windows 10/11 64-bit, 8 GB RAM or more
recommended. SmartScreen may warn because the build is unsigned:
"More info" then "Run anyway".
