# Docker: Index-Watcher

Dieser Stack startet PostgreSQL (pgvector), Apache Tika und den
Datei-Watcher als Docker-Container.

Der Watcher **überwacht nur** — Tabellen, Datapool und Erstindexierung müssen
vorher manuell angelegt werden.

## Voraussetzungen

```bash
meipi-index create-tables
meipi-index create-pool --name default --rootpath /data
meipi-index read-files --pool-id 1 .
```

Der `rootpath` des Pools muss dem Mount-Pfad im Container entsprechen (Standard: `/data`).

## Schnellstart

```bash
cp docker/.env.example docker/.env
# IND_WATCH_POOL_ID, HOST_DATA_DIR und Passwörter anpassen

docker compose --env-file docker/.env up --build -d
docker compose logs -f watcher
```

Stoppen:

```bash
docker compose down
```

## Volumes

| Host | Container | Zweck |
|------|-----------|-------|
| `HOST_DATA_DIR` (Standard: `./data`) | `IND_WATCH_POOL_ROOTPATH` (Standard: `/data`) | Zu überwachende Dateien |

## Wichtige Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `IND_WATCH_POOL_ID` | — | **Pflicht.** Id eines bestehenden Datapools |
| `IND_WATCH_POOL_ROOTPATH` | `/data` | Mount-Ziel; muss `rootpath` des Pools sein |
| `IND_WATCH_PATH` | `.` | Relativer Pfad unterhalb des Pools |
| `IND_WATCH_DEBOUNCE` | `1.0` | Sekunden bis zur Indexierung nach Änderung |
| `IND_WATCH_NO_THUMBS` | `0` | Thumbnails per PIL deaktivieren |

## Nur Watcher-Image bauen

```bash
docker build -t meipi-index-watcher .
docker run --rm \
  -e IND_PG_HOST=host.docker.internal \
  -e IND_PG_PASSWD=secret \
  -e IND_TIKA_NOOCR_URL=http://host.docker.internal:9998 \
  -e IND_WATCH_POOL_ID=1 \
  -v /pfad/zu/dateien:/data:ro \
  meipi-index-watcher
```

## Hinweise

- Apache Tika läuft mit `docker/tika-config.xml` (OCR-Parser deaktiviert). Der Watcher nutzt
  `IND_TIKA_NOOCR_URL` auf Port 9998.
- Einzelbild-Thumbnails laufen per PIL (kein CUDA/DALI nötig).
- Der Keyring wird im Container nicht genutzt; setzen Sie `IND_PG_PASSWD` direkt.
- Für Produktion: sichere Passwörter, persistente Volumes und ggf. separates Tika/OCR.
