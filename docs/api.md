# SpeechInt – API-Vertrag

Diese Datei ist der eingefrorene Vertrag zwischen SpeechInt und seinen Clients
(vor allem `dareinelt/LLMInt`). Sie beschreibt Endpunkte, Nutzlasten, Fehlercodes
und den Ladevertrag. Änderungen hier sind Schnittstellenänderungen.

- Basis-URL: `http://<host>:<port>` (Standardport 8080 im Container, siehe
  `SPEECHINT_PORT`)
- Authentifizierung: `X-Auth-Token: <token>` **oder**
  `Authorization: Bearer <token>`. Ist auf der Serverseite kein Token gesetzt,
  sind alle geschützten Routen offen.
- Zeichensatz: UTF-8, Antworten sind JSON (Ausnahme: `response_format=text`)
- Kein Streaming, keine Sitzungen: der Dienst ist zustandslos. Der Client hält
  den diktierten Text und schickt bei Bedarf nur das neue Fragment.

## Endpunkte im Überblick

| Methode | Pfad | Auth | Zweck |
|---|---|---|---|
| GET | `/health` | nein | Liveness des Gateways (Container-Healthcheck) |
| GET | `/v1/ready` | nein | Bereitschaft: 200 bereit / 503 lädt |
| GET | `/v1/health` | ja | Zustand beider Modellserver (200/503) |
| GET | `/v1/config` | ja | Fähigkeiten, Vorgaben, Grenzen, Host-Dimensionierung |
| GET | `/v1/models` | ja | Ausgelieferte Modelle im OpenAI-`list`-Umschlag |
| POST | `/v1/audio/transcriptions` | ja | Audiosegment → Rohtranskript |
| POST | `/v1/chat/completions` | ja | OpenAI-kompatibler Durchgriff auf llama.cpp |
| POST | `/v1/dictate/process` | ja | Transkriptfragment → fertiger Text |
| POST | `/v1/dictate` | ja | Audio → fertiger Text in einem Aufruf |

## Fehlerumschlag

Jede fehlgeschlagene Anfrage antwortet mit demselben Umschlag:

```json
{ "ok": false, "error": "<code>", "message": "<deutscher Text>" }
```

`error` ist der stabile Code, auf den ein Client verzweigen kann; `message` ist
für die Anzeige im Adminbereich gedacht.

| HTTP | `error` | Bedeutung |
|---|---|---|
| 400 | `bad_request` | Anfrage nicht lesbar (fehlendes Feld, kein JSON-Objekt) |
| 401 | `unauthorized` | Token fehlt oder passt nicht |
| 413 | `payload_too_large` | Aufnahme größer als `max_audio_bytes` (Standard 25 MB) |
| 502 | `whisper_unreachable`, `whisper_failed`, `model_unreachable`, `model_failed` | Modellserver antwortet nicht oder fehlerhaft |
| 503 | `service_loading` | Modell wird noch heruntergeladen/geladen – **bitte erneut versuchen** |
| 503 | `not_configured` | Für die Komponente ist keine URL konfiguriert |
| 504 | `whisper_timeout`, `model_timeout` | Modellserver hat nicht rechtzeitig geantwortet |

`service_loading` trägt zusätzlich `component` (`whisper` oder `llm`) und
`retry_after` (Sekunden) und setzt immer den Header `Retry-After`.

## Ladevertrag

Der erste Start des Stapels ist langsam: whisper.cpp lädt ein GGML-Modell,
llama.cpp zieht ein ~1,1 GB großes GGUF. Damit ein Client das von einem Defekt
unterscheiden kann, melden beide Zustandsrouten dasselbe:

```json
{
  "ok": false,
  "service": "speechint",
  "status": "loading",
  "ready": false,
  "message": "Noch nicht bereit – die Modelle werden geladen oder heruntergeladen. Spracherkennung (Whisper): startet; Diktat-Modell (Qwen): bereit. Bitte in 15 s erneut versuchen.",
  "retry_after": 15,
  "components": {
    "whisper": {
      "state": "starting",
      "state_label": "startet",
      "ok": false,
      "http": 0,
      "message": "Whisper nicht erreichbar: All connection attempts failed",
      "url": "http://whisper:8080",
      "model": "small"
    },
    "llm": {
      "state": "ready",
      "state_label": "bereit",
      "ok": true,
      "http": 200,
      "message": "Modellserver erreichbar (ok).",
      "url": "http://llm:8080",
      "model": "Qwen3.5-2B Q4"
    }
  }
}
```

`status` ist `ready`, `loading`, `degraded` oder `unavailable`.

| Zustand einer Komponente | Bedeutung |
|---|---|
| `ready` | Server antwortet, Modell geladen |
| `loading` | Server antwortet und meldet selbst „Modell lädt“ |
| `starting` | noch nie erreichbar und innerhalb der Kulanzzeit (`SPEECHINT_STARTING_GRACE_SECONDS`, Standard 900 s) – wird heruntergeladen/geladen |
| `unreachable` | war schon einmal bereit (oder Kulanzzeit abgelaufen) und ist jetzt weg |
| `error` | erreichbar, antwortet aber unerwartet |
| `unconfigured` | keine URL konfiguriert |

Empfohlenes Client-Verhalten (so macht es LLMInt):

1. Nach dem Start und während einer Aufnahme `GET /v1/ready` pollen (ohne Token
   nutzbar) und `message` anzeigen.
2. Bei 503 `Retry-After` abwarten, nicht sofort erneut senden.
3. Nur `loading`/`starting` als „bitte warten“ behandeln. Ist eine Komponente
   `unreachable`/`error`/`unconfigured`, ist das ein echter Fehler bzw. der
   Regel-Fallback der Diktat-Pipeline greift.

Die Antworten von `/v1/ready` und `/v1/health` sind identisch; `/v1/ready` ist
für Orchestrierung ohne Token gedacht, `/v1/health` für den Client mit Token.
Proben werden serverseitig für `SPEECHINT_HEALTH_CACHE_SECONDS` (Standard 5 s)
zwischengespeichert – häufiges Pollen ist billig.

## POST /v1/audio/transcriptions

`multipart/form-data`

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `file` | Datei | ja | Audiosegment (wav, webm, ogg, mp3, …); whisper.cpp wandelt intern um |
| `language` | Text | nein | ISO-Sprache, Standard aus der Konfiguration (`de`) |
| `model` | Text | nein | Modell überschreiben; leer = konfiguriertes Modell (`small`) |
| `response_format` | Text | nein | `json` (Standard) oder `text` |

Antwort 200 (`json`):

```json
{
  "ok": true,
  "text": "hallo welt",
  "empty": false,
  "model": "small",
  "language": "de",
  "duration_ms": 412,
  "bytes": 33152,
  "total_ms": 430
}
```

Ein leeres Segment ist kein Fehler: die Antwort enthält `"text": ""`,
`"empty": true`, ohne den Whisper-Server zu belasten.

## POST /v1/dictate/process

`application/json`

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `fragment` | Text | ja | neu erkanntes Fragment (ein Aufnahmesegment) |
| `context` | Text | nein | bereits diktierter Text, nur lesbar; wird serverseitig auf `max_context_chars` gekürzt |
| `prompt` | Text | nein | Systemprompt des Clients; leer = Vorgabeprompt |
| `commands` | Liste | nein | Befehlstabelle des Clients (`[{"phrase": "neue zeile", "replacement": "\n"}]`); leer = Vorgabetabelle |
| `model` | Text | nein | Modell überschreiben |
| `temperature` | Zahl | nein | Standard 0.1 |
| `max_tokens` | Ganzzahl | nein | Standard abgeleitet aus der Fragmentlänge |

Antwort 200:

```json
{
  "ok": true,
  "text": "Hallo Welt.",
  "fallback": false,
  "model": "Qwen3.5-2B Q4",
  "warning": "",
  "duration_ms": 268
}
```

- `fallback: true` heißt: das Modell war nicht nutzbar, `text` ist die
  regelbasierte Aufbereitung des Fragments (`warning` nennt den Grund). Der
  erkannte Text geht also nie verloren.
- Ist das Modell noch im Ladezustand (`loading`/`starting`), antwortet die Route
  stattdessen mit 503 `service_loading` – der Client behält das Fragment und
  versucht es erneut, statt auf den Regel-Fallback auszuweichen.
- Ein leeres Fragment liefert `text: ""`, `fallback: false`.

Der Client bleibt Eigentümer von Prompt, Befehlstabelle und Kontext; der Dienst
ist zustandslos und speichert weder Text noch Audio.

## POST /v1/dictate

`multipart/form-data` – Transkription und Nachbearbeitung in einem Aufruf.

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `file` | Datei | ja | Audiosegment |
| `language` | Text | nein | wie oben |
| `context` | Text | nein | wie oben |
| `prompt` | Text | nein | wie oben |
| `commands` | Text | nein | Befehlstabelle als **JSON-String** (Multipart) |
| `model` | Text | nein | Modell überschreiben |

Antwort 200:

```json
{
  "ok": true,
  "text": "Hallo Welt.",
  "transcript": "hallo welt",
  "fallback": false,
  "model": "Qwen3.5-2B Q4",
  "warning": "",
  "whisper_model": "small",
  "transcribe_ms": 412,
  "process_ms": 268,
  "total_ms": 690
}
```

Beide Komponenten werden vorab geprüft: ist eine noch im Ladezustand, kommt 503
`service_loading`, ohne dass transkribiert wird.

## POST /v1/chat/completions

OpenAI-kompatibler Durchgriff auf llama.cpp. Der Körper wird unverändert
weitergereicht; nur `stream` wird auf `false` gezwungen und fehlende
Diktat-Vorgaben (`chat_template_kwargs.enable_thinking = false`,
`reasoning_budget = 0`) werden ergänzt. Antwort ist die Rohantwort von
llama.cpp.

## GET /v1/config

```json
{
  "ok": true,
  "service": "speechint",
  "version": "1.0.0",
  "capabilities": ["transcribe", "process", "dictate", "chat"],
  "transcription": { "configured": true, "model": "small", "language": "de", "timeout_seconds": 120.0 },
  "dictation": {
    "configured": true,
    "model": "Qwen3.5-2B Q4",
    "timeout_seconds": 60.0,
    "max_context_chars": 400,
    "temperature": 0.1,
    "commands": [{ "phrase": "neue zeile", "replacement": "\n" }],
    "default_prompt": "…"
  },
  "limits": { "max_audio_bytes": 26214400, "max_audio_mb": 25.0, "loading_retry_after": 15 },
  "host": {
    "cores": 6,
    "memory_gb": 16.0,
    "container_memory_gb": 0.5,
    "avx2": true,
    "architecture": "x86_64",
    "minimum": { "cores": 6, "memory_gb": 16.0, "avx2": true },
    "budget_gb": { "gateway": 0.3, "whisper": 1.2, "llm": 3.0 },
    "meets_minimum": true,
    "warnings": []
  }
}
```

Ein Client nutzt das, um im Adminbereich Vorgaben (Prompt, Befehlstabelle) und
Modellnamen vorzubelegen und um die Eignung des Endpunkts anzuzeigen
(`host.meets_minimum`, `host.warnings`). `host.memory_gb` ist der Speicher der
Maschine, auf der die Modelle laufen; `host.container_memory_gb` ist die
(absichtlich kleine) Grenze des Gateway-Containers.

## Mindestanforderung an einen Speech-Endpunkt

Mindestens **6 CPU-Kerne mit AVX2** und **mindestens 16 GB RAM** (siehe
`README.md`). Der Dienst läuft auch darunter, meldet die Unterschreitung aber in
`host.warnings` und im Startprotokoll.
