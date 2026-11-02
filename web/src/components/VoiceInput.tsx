// Voice input: records with MediaRecorder, sends the raw clip for transcription
// and hands the editable text to the parent. It never sends the chat itself.
import { useEffect, useRef, useState } from "react";
import { transcribeAudio } from "../api";
import Icon from "./Icon";

export interface VoiceInputProps {
  lang: "en" | "zh";
  disabled: boolean;
  onTranscript: (text: string) => void;
  onBusyChange?: (busy: boolean) => void;
}

type Status = "idle" | "recording" | "transcribing" | "review" | "error";
type ErrorKind = "permission" | "unavailable" | "unsupported" | "silence" | "tooLong" | "busy";

const MAX_SECONDS = 30;
const MAX_BYTES = 8 * 1024 * 1024;
const MIME_TYPES = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"];

const STRINGS = {
  en: {
    start: "Start voice input",
    stop: "Stop recording",
    transcribingLabel: "Transcribing voice input",
    recording: "Recording… tap to stop (30 seconds maximum).",
    silence: "No speech detected. Please try again.",
    tooLong: "Please record a shorter message (up to 30 seconds).",
    busy: "Voice recognition is busy. Please retry shortly.",
    transcribing: "Transcribing…",
    review: "Transcript ready. Edit it before sending.",
    permission: "Microphone access was denied. Allow it in your browser settings, then retry.",
    unavailable: "Voice input is unavailable right now. Please type instead.",
    unsupported: "Voice input needs a modern browser over HTTPS (or localhost).",
    retry: "Retry",
  },
  zh: {
    start: "开始语音输入",
    stop: "停止录音",
    transcribingLabel: "正在转写语音",
    recording: "正在录音，点击停止（最长30秒）。",
    silence: "未检测到语音，请重试。",
    tooLong: "请缩短语音，最长30秒。",
    busy: "语音识别正忙，请稍后重试。",
    transcribing: "正在转写…",
    review: "转写完成，发送前可编辑。",
    permission: "麦克风权限被拒绝，请在浏览器设置中允许后重试。",
    unavailable: "语音输入暂不可用，请改用文字输入。",
    unsupported: "语音输入需要支持录音的现代浏览器，并通过 HTTPS（或 localhost）打开。",
    retry: "重试",
  },
} as const;

export default function VoiceInput({ lang, disabled, onTranscript, onBusyChange }: VoiceInputProps) {
  const text = STRINGS[lang];
  const [status, setStatus] = useState<Status>("idle");
  const [errorKind, setErrorKind] = useState<ErrorKind | null>(null);

  const disposed = useRef(false);
  const generation = useRef(0);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const bytes = useRef(0);
  const timer = useRef<number | null>(null);
  const upload = useRef<AbortController | null>(null);
  const stopRequested = useRef(false);
  const transcript = useRef(onTranscript);
  const busyChange = useRef(onBusyChange);
  const busyValue = useRef(false);
  transcript.current = onTranscript;
  busyChange.current = onBusyChange;

  function clearTimer() {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }

  function stopTracks(target: MediaStream | null) {
    target?.getTracks().forEach((track) => track.stop());
  }

  // Stop the stream, timer and any live recorder.
  function teardown() {
    clearTimer();
    const active = recorder.current;
    recorder.current = null;
    if (active && active.state !== "inactive") {
      try {
        active.stop();
      } catch {
        /* already stopped */
      }
    }
    stopTracks(stream.current);
    stream.current = null;
    stopRequested.current = false;
  }

  // Invalidate pending callbacks, abort any upload and drop the recording.
  function release() {
    generation.current += 1;
    upload.current?.abort();
    upload.current = null;
    chunks.current = [];
    bytes.current = 0;
    teardown();
  }

  function fail(kind: ErrorKind) {
    release();
    setErrorKind(kind);
    setStatus("error");
  }

  function cancelAll() {
    release();
    setErrorKind(null);
    setStatus("idle");
  }

  async function send(blob: Blob, run: number) {
    setStatus("transcribing");
    const controller = new AbortController();
    upload.current = controller;
    try {
      const result = await transcribeAudio(blob, controller.signal);
      if (disposed.current || run !== generation.current || controller.signal.aborted) return;
      const value = typeof result?.text === "string" ? result.text.trim() : "";
      if (!value) {
        setErrorKind("unavailable");
        setStatus("error");
        return;
      }
      setStatus("review");
      transcript.current(value);
    } catch (error) {
      if (disposed.current || run !== generation.current || controller.signal.aborted) return;
      const code = error instanceof Error ? error.message : "";
      setErrorKind(code === "no_speech" ? "silence" : code === "audio_too_long" ? "tooLong" : code === "speech_busy" ? "busy" : "unavailable");
      setStatus("error");
    } finally {
      if (upload.current === controller) upload.current = null;
    }
  }

  async function start() {
    if (disabled || disposed.current || status === "recording" || status === "transcribing") return;
    const run = (generation.current += 1);
    chunks.current = [];
    bytes.current = 0;
    stopRequested.current = false;
    setErrorKind(null);

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined" || !window.isSecureContext) {
      setErrorKind("unsupported");
      setStatus("error");
      return;
    }

    setStatus("recording");
    let media: MediaStream | null = null;
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      if (disposed.current || run !== generation.current) return;
      const name = (error as { name?: string } | null)?.name ?? "";
      const denied = ["NotAllowedError", "PermissionDeniedError", "SecurityError"].includes(name);
      setErrorKind(denied ? "permission" : "unavailable");
      setStatus("error");
      return;
    }
    // Permission resolved after stop/unmount: drop the tracks immediately.
    if (disposed.current || run !== generation.current) {
      stopTracks(media);
      return;
    }
    if (!media) return;
    stream.current = media;

    const mimeType = MIME_TYPES.find(
      (type) => typeof MediaRecorder.isTypeSupported === "function" && MediaRecorder.isTypeSupported(type),
    );
    let active: MediaRecorder;
    try {
      active = mimeType ? new MediaRecorder(media, { mimeType }) : new MediaRecorder(media);
    } catch {
      fail("unavailable");
      return;
    }
    recorder.current = active;

    active.ondataavailable = (event) => {
      if (disposed.current || run !== generation.current) return;
      if (!event.data || event.data.size === 0) return;
      bytes.current += event.data.size;
      if (bytes.current > MAX_BYTES) {
        fail("unavailable");
        return;
      }
      chunks.current.push(event.data);
    };
    active.onerror = () => {
      if (!disposed.current && run === generation.current) fail("unavailable");
    };
    active.onstop = () => {
      if (disposed.current || run !== generation.current) return;
      const byUser = stopRequested.current;
      const type = active.mimeType || mimeType || "audio/webm";
      const collected = chunks.current;
      chunks.current = [];
      bytes.current = 0;
      teardown();
      // Only a user stop or the 30 s cap may submit; discard unexpected stops.
      if (!byUser || collected.length === 0) {
        setErrorKind("unavailable");
        setStatus("error");
        return;
      }
      void send(new Blob(collected, { type }), run);
    };

    try {
      active.start(1000);
    } catch {
      fail("unavailable");
      return;
    }
    timer.current = window.setTimeout(stop, MAX_SECONDS * 1000);
  }

  function stop() {
    if (stopRequested.current) return;
    stopRequested.current = true;
    clearTimer();
    const active = recorder.current;
    // Permission still pending: cancel it; late resolution stops its tracks.
    if (!active || active.state === "inactive") {
      release();
      setStatus("idle");
      return;
    }
    try {
      active.stop();
    } catch {
      fail("unavailable");
    }
  }

  function toggle() {
    if (disabled || status === "transcribing") return;
    if (status === "recording") stop();
    else void start();
  }

  useEffect(() => {
    const busy = status === "recording" || status === "transcribing";
    if (busyValue.current !== busy) {
      busyValue.current = busy;
      busyChange.current?.(busy);
    }
  }, [status]);

  useEffect(() => {
    if (disabled) cancelAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabled]);

  useEffect(() => {
    disposed.current = false;
    return () => {
      disposed.current = true;
      release();
      if (busyValue.current) {
        busyValue.current = false;
        busyChange.current?.(false);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const recording = status === "recording";
  const label = recording ? text.stop : status === "transcribing" ? text.transcribingLabel : text.start;
  const message =
    status === "recording"
      ? text.recording
      : status === "transcribing"
        ? text.transcribing
        : status === "review"
          ? text.review
          : "";

  return (
    <div style={{ position: "relative", display: "flex", flexDirection: "column", gap: "0.3rem", minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <button
          type="button"
          className="icon-btn"
          onClick={toggle}
          disabled={disabled || status === "transcribing"}
          aria-label={label}
          aria-pressed={recording}
          style={{
            width: "3.1rem",
            height: "3.1rem",
            color: recording ? "var(--danger-dark)" : "var(--primary)",
            background: recording ? "var(--danger-soft)" : undefined,
          }}
        >
          <Icon name="mic" size={22} />
        </button>
        {status === "error" && (
          <button type="button" className="pill-btn" onClick={() => void start()} disabled={disabled}>
            {text.retry}
          </button>
        )}
      </div>
      {message && (
        <div aria-live="polite" className="small muted" style={{ position: "absolute", bottom: "calc(100% + 0.5rem)", right: 0, width: "max-content", maxWidth: "min(70vw, 20rem)", padding: "0.5rem", background: "#fff", borderRadius: "0.5rem" }}>
          {message}
        </div>
      )}
      {status === "error" && errorKind && (
        <div role="alert" className="small" style={{ position: "absolute", bottom: "calc(100% + 0.5rem)", right: 0, width: "max-content", maxWidth: "min(70vw, 20rem)", padding: "0.5rem", background: "#fff", borderRadius: "0.5rem", color: "var(--danger-dark)" }}>
          {text[errorKind]}
        </div>
      )}
    </div>
  );
}
