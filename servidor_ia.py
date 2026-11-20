"""Backend robot. Python 3 standard library only. Behind managed HTTPS proxy."""

import base64
import os
import json
import urllib.request
import io
import wave
import struct
import secrets
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = os.environ["GEMINI_API_KEY"]
TOKEN = os.environ["ROBOT_TOKEN"]

if len(TOKEN) < 32:
    raise RuntimeError("ROBOT_TOKEN requiere 32 caracteres o mas")

LOCK = threading.Lock()
DAY = ""
COUNT = 0
LIMIT = int(os.getenv("MAX_TURNS_PER_DAY", "30"))
HISTORY = []


def api(model, body):
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-._"
    )
    if not model or any(c not in allowed for c in model):
        raise ValueError("modelo invalido")

    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model
        + ":generateContent",
        data=json.dumps(body).encode(),
        headers={
            "x-goog-api-key": KEY,
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=18) as r:
        data = r.read(3000001)

    if len(data) > 3000000:
        raise ValueError("respuesta excesiva")

    return json.loads(data)


def parts(response):
    return response["candidates"][0]["content"]["parts"]


def turn(raw):
    mono = b"".join(raw[i:i + 2] for i in range(0, len(raw), 4))

    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(mono)

    user = {
        "role": "user",
        "parts": [
            {
                "text": (
                    "Responde a lo que digo en esta grabacion. "
                    "Si no hay palabras claras, pide que repita."
                )
            },
            {
                "inlineData": {
                    "mimeType": "audio/wav",
                    "data": base64.b64encode(wav.getvalue()).decode(),
                }
            },
        ],
    }

    response = api(
        os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
        {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "Eres Juana, una asistente robot de escritorio "
                            "amable, curiosa y con humor suave. "
                            "Habla en español en una o dos frases, "
                            "máximo 35 palabras. "
                            "No puedes enviar mensajes, buscar en internet, "
                            "guardar recordatorios ni mover motores. "
                            "No finjas haberlo hecho. "
                            "Tu voz es sintetizada por IA."
                        )
                    }
                ]
            },
            "contents": HISTORY + [user],
            "generationConfig": {
                "maxOutputTokens": 512,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
    )

    answer = "".join(
        p.get("text", "")
        for p in parts(response)
        if not p.get("thought")
    )[:350].strip()

    if not answer:
        raise ValueError("sin respuesta")

    speech = api(
        os.getenv("TTS_MODEL", "gemini-3.8-flash-tts"),
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": answer,
                            "speech_metadata": {
                                "style": (
                                    "Warm, friendly conversational Spanish. "
                                    "Natural pace."
                                )
                            },
                        }
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "voice": os.getenv("GEMINI_VOICE", "Aoede")
                    }
                },
            },
        },
    )

    audio = next(
        p["inlineData"]
        for p in parts(speech)
        if "inlineData" in p
    )
    encoded = base64.b64decode(audio["data"], validate=True)

    # El servidor espera audio WAV y rechaza formatos inesperados.
    with wave.open(io.BytesIO(encoded), "rb") as w:
        if (
            w.getnchannels() != 1
            or w.getsampwidth() != 2
            or w.getcomptype() != "NONE"
        ):
            raise ValueError("formato de voz inesperado")

        rate = w.getframerate()

        if not 8000 <= rate <= 48000 or w.getnframes() > rate * 20:
            raise ValueError("respuesta de voz demasiado larga")

        pcm = w.readframes(w.getnframes())

    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)

    if not samples:
        raise ValueError("voz vacia")

    out = bytearray()

    for k in range(len(samples) * 16000 // rate):
        pos = k * rate / 16000
        i = int(pos)
        f = pos - i
        v = int(
            samples[i] * (1 - f)
            + samples[min(i + 1, len(samples) - 1)] * f
        )
        out.extend(struct.pack("<hh", v, v))

    # Solo guarda respuestas previas de Juana en RAM.
    # No guarda las grabaciones del usuario.
    HISTORY.extend(
        [
            {
                "role": "user",
                "parts": [{"text": "Continuemos la conversación."}],
            },
            {
                "role": "model",
                "parts": [{"text": answer}],
            },
        ]
    )
    del HISTORY[:-8]

    return bytes(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # No registrar audio, textos ni tokens.

    def reply(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def auth(self):
        return secrets.compare_digest(
            self.headers.get("Authorization", ""),
            "Bearer " + TOKEN,
        )

    def do_GET(self):
        # Comprobación pública: no revela secretos ni llama a Gemini.
        if self.path == "/ready":
            self.reply(200, b"Juana ready")
        else:
            self.reply(
                200 if self.path == "/health" and self.auth() else 403
            )

    def do_POST(self):
        global DAY, COUNT

        self.connection.settimeout(10)

        if self.path != "/turn" or not self.auth():
            self.reply(403)
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.reply(400)
            return

        if (
            not 16000 <= n <= 384000
            or n % 4
            or self.headers.get("Transfer-Encoding")
        ):
            self.reply(413)
            return

        if not LOCK.acquire(False):
            self.reply(429)
            return

        try:
            day = time.strftime("%Y-%m-%d", time.gmtime())

            if DAY != day:
                DAY = day
                COUNT = 0

            if COUNT >= LIMIT:
                self.reply(429)
                return

            # Las solicitudes fallidas también consumen la cuota.
            COUNT += 1

            raw = self.rfile.read(n)
            if len(raw) != n:
                raise ValueError("incompleto")

            self.reply(200, turn(raw))

        except Exception:
            self.reply(502)
        finally:
            LOCK.release()


if __name__ == "__main__":
    ThreadingHTTPServer(
        (
            os.getenv("HOST", "127.0.0.1"),
            int(os.getenv("PORT", "8080")),
        ),
        Handler,
    ).serve_forever()
