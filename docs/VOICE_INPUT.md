# Local voice input

Voice input uses Qwen3-ASR-1.7B, with Apache-2.0 licensed weights. It is a
separate speech model; the r3 banking planner is unchanged. The official model
supports Chinese, English and other languages and dialects. Published comparisons
are the authors' benchmarks, not a guarantee that it is best for every speaker.
This implementation does not claim to use ChatGPT's internal speech model.

Sources: [official model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
[official implementation](https://github.com/QwenLM/Qwen3-ASR).

## What it does

Browser MediaRecorder sends audio to the authenticated `/voice/transcribe`
endpoint, which calls the loopback ASR service. The editable message then goes
through normal tokenization, planning, policy and approval.

The browser needs HTTPS or localhost. A plain HTTP LAN address generally cannot
access the microphone. The local UI and tunnel must remain running.

With an empty message box, tap the microphone and allow browser access. Speak,
then tap again to stop (automatic stop at 30 s). Review and edit the transcript,
especially names and amounts, before pressing Send. Clear the message box to
record again. Transcription never sends chat, approves a mandate or treats
spoken "yes" as verification.

English cardinal amounts with explicit yuan/RMB currency are tokenized. The
spoken variant "Lena" can match the saved demo contact "李娜 Lina" when there is
only one such contact and no exact Lena contact. Exact contact matches take
precedence. No general fuzzy recipient selection was added.

The exact transcript "four hundred fifty yuan? No, no, no, five hundred fifty
yuan to Lena" was checked against the live r3 planner. It proposes 550 yuan to
李娜 Lina···1660 and waits for confirmation; the account balance is unchanged.
Both spoken amounts and correction words remain in the planner input. The full
regression suite after this fix passes 1,270 tests. This does not establish that
every correction or similar-sounding name will be understood.

## Run the ASR service

Use a separate Python environment for speech: `qwen-asr==0.0.6` requires
Transformers 4.57.6, while the planner training environment has a different
version. Install the appropriate CUDA PyTorch build in the speech environment,
then `pip install -r requirements-voice.txt`. Download the official model into
a new local directory using ModelScope or Hugging Face as documented upstream.
The serving process requires an existing local model path.

```bash
# Speech environment, on an idle GPU:
export CUDA_VISIBLE_DEVICES=2
export ANBAN_ASR_MODEL=/absolute/path/to/Qwen3-ASR-1.7B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m api.asr_service  # loopback 18090
```

```bash
# Application environment, separate terminal:
export ANBAN_ASR_URL=http://127.0.0.1:18090
export ANBAN_DEMO=1 ANBAN_PLANNER=local ANBAN_PLANNER_PROFILE=r3
export ANBAN_LOCAL_URL=http://127.0.0.1:18081/v1 ANBAN_LOCAL_MODEL=anban-r3
unset ANBAN_FALLBACK_PLANNER
python -m uvicorn api.app:create_app --factory --host 127.0.0.1 --port 18000
```

The demo login endpoints are for the local mock demo only.

```bash
# Client machine: keep this tunnel open; do not expose the ASR port publicly.
ssh -N -L 127.0.0.1:18000:127.0.0.1:18000 -p YOUR_SSH_PORT YOUR_USER@YOUR_HOST
```

```powershell
# Client PowerShell, after npm ci in web:
$env:ANBAN_API_TARGET='http://127.0.0.1:18000'
npm.cmd --prefix web run dev -- --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174/` after the dev server starts.

## Limits and behaviour

- Up to 30 seconds and 8 MiB per recording; transcripts up to 500 characters.
- Language is detected from audio, independently of the UI language setting.
- WebM/Opus, Ogg/Opus, MP4/AAC and WAV inputs are supported. No browser speech
  service or paid transcription API is used as a fallback.
- The original audio is processed in server memory. The service does not save
  recordings or log their content. Speech recognition necessarily sees the
  original speech; semantic masking happens later, before the banking planner.
- Audio decoding uses container/protocol allowlists and a decoded-duration cap.
  Silence is rejected. The model has one inference slot and reports busy rather
  than accumulating GPU jobs. The proxy forwards no banking bearer or cookies.
- Recordings and uploads are cancelled when leaving the component or changing
  persona. Late permission results release their microphone tracks.
- ASR is transcription, not sarcasm detection, identity verification or proof
  that an amount was heard correctly. Manual review and banking gates remain.

## Verification

- Full Python suite: 1,188 passed, including 35 voice-specific checks.
- Five DOM tests drive the actual recorder component: editable transcript,
  denied permission, late permission after unmount, upload cancellation and
  disabling during recording. Run `npm --prefix web run test:voice` after
  `npm ci`; the locked jsdom dev dependency needs Node 20.19+, 22.13+ or 24+.
- TypeScript and production Vite build passed.
- Actual Qwen inference through the authenticated bank API transcribed the
  upstream Chinese and English reference audio. The first request, including
  model loading, took 13.39 s; the second took 1.89 s. These are smoke timings.
- Actual encoded WebM/Opus, Ogg/Opus and MP4/AAC reference clips all returned 200
  with the same Chinese transcript. Bank transactions were unchanged.
- A physical microphone/browser permission flow still needs a user check on
  the target device. Dialect, noise and banking name/amount accuracy have not
  been benchmarked by these smoke tests.

The speech service uses about 4.8 GiB of GPU memory. The r3 planner runs in a
separate environment.
