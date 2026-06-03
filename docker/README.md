# Docker: Index-Watcher

Dieser Stack startet PostgreSQL (pgvector), Apache Tika und den
Datei-Watcher als Docker-Container.

## Schnellstart

```bash
# 1. Datenverzeichnis anlegen (Beispiel)
mkdir -p data

# 2. Umgebung konfigurieren
cp docker/.env.example docker/.env
# HOST_DATA_DIR und Passwörter bei Bedarf anpassen

# 3. Stack starten
docker compose --env-file docker/.env up --build -d
```

Der Watcher:

1. legt Tabellen an (`MEIPI_ENSURE_TABLES=1`)
2. registriert einen Datapool (`MEIPI_POOL_NAME` / `MEIPI_POOL_ROOTPATH`)
3. führt optional einen Initial-Scan aus (`MEIPI_INITIAL_SCAN=1`)
4. überwacht den gemounteten Ordner dauerhaft

Logs ansehen:

```bash
docker compose logs -f watcher
```

Stoppen:

```bash
docker compose down
```

## Volumes

| Host | Container | Zweck |
|------|-----------|-------|
| `HOST_DATA_DIR` (Standard: `./data`) | `/data` | Zu indexierende Dateien |

Der `rootpath` des Datapools muss mit dem Mount-Pfad im Container übereinstimmen
(Standard: `/data`).

## Wichtige Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `MEIPI_POOL_NAME` | `default` | Name des Datapools |
| `MEIPI_POOL_ROOTPATH` | `/data` | Root-Pfad im Container |
| `MEIPI_WATCH_PATH` | `.` | Relativer Pfad unterhalb des Pools |
| `MEIPI_INITIAL_SCAN` | `1` | Einmaliger Vollscan vor dem Watch |
| `MEIPI_DEBOUNCE` | `1.0` | Sekunden bis zur Indexierung nach Änderung |
| `MEIPI_NO_THUMBS` | `1` | Thumbnails deaktiviert (CPU-Image) |
| `MEIPI_ENSURE_TABLES` | `1` | `create-tables` beim Start |

Alternativ kann statt Pool-Auto-Erstellung `MEIPI_POOL_ID` gesetzt werden, wenn der
Pool bereits existiert.

## Nur Watcher-Image bauen

```bash
docker build -t meipi-index-watcher .
docker run --rm \
  -e IND_PG_HOST=host.docker.internal \
  -e IND_PG_PASSWD=secret \
  -e IND_TIKA_NOOCR_URL=http://host.docker.internal:9998 \
  -e MEIPI_POOL_ID=1 \
  -e MEIPI_NO_THUMBS=1 \
  -v /pfad/zu/dateien:/data:ro \
  meipi-index-watcher
```

## Hinweise

- Thumbnails benötigen CUDA/DALI und sind in diesem schlanken Image standardmäßig
  ausgeschaltet (`--no-thumbs`). Metadaten und Volltext werden trotzdem indexiert.
- Der Keyring wird im Container nicht genutzt; setzen Sie `IND_PG_PASSWD` direkt.
- Für Produktion: sichere Passwörter, persistente Volumes und ggf. separates Tika/OCR.
