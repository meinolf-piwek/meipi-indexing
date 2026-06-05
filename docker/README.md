# Docker: Index-Watcher

Dieser Stack startet Apache Tika und den Datei-Watcher. PostgreSQL läuft
extern (Container `pg-db` im Docker-Netzwerk `postgresql_default`).

Beim Start prüft der Watcher, ob Datenbank und Dateisystem übereinstimmen, und
meldet Schema-Informationen sowie Abweichungen (ohne sie zu beheben). Derselbe
Check ist separat verfügbar: ``meipi-index check-sync --pool-id 1 .``
Tabellen, Datapool und Erstindexierung müssen vorher angelegt werden.

## Voraussetzungen

- PostgreSQL mit pgvector erreichbar als `pg-db` im Netzwerk `postgresql_default`
- Netzwerk muss existieren: `docker network ls | grep postgresql_default`

```bash
export IND_DOCROOT=/pfad/zu/dateien   # oder: meipi-index --docroot /pfad/zu/dateien …
meipi-index create-tables
meipi-index create-pool --name default
meipi-index read-files --pool-id 1 .
```

Im Container: `IND_DOCROOT=/data` und Volume-Mount auf denselben Pfad.

## Schnellstart

```bash
cp docker/.env.example docker/.env
# IND_PG_*, IND_DOCROOT, IND_WATCH_POOL_ID, HOST_DATA_DIR anpassen

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
| `HOST_DATA_DIR` (Standard: `./data`) | `IND_DOCROOT` (Standard: `/data`) | Zu überwachende Dateien |

## Wichtige Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `IND_DOCROOT` | `/data` | Dateisystem-Root; Pfade in der DB sind relativ dazu |
| `IND_PG_HOST` | `pg-db` | PostgreSQL-Hostname |
| `IND_PG_USER` / `IND_PG_PASSWD` / `IND_PG_DATABASE` | — | DB-Zugangsdaten |
| `IND_WATCH_POOL_ID` | — | **Pflicht.** Id eines bestehenden Datapools |
| `IND_WATCH_PATH` | `.` | Path under `IND_DOCROOT` (relative, e.g. `docs`, or absolute under docroot) |
| `IND_WATCH_DEBOUNCE` | `1.0` | Sekunden bis zur Indexierung nach Änderung |
| `IND_WATCH_NO_THUMBS` | `0` | Thumbnails per PIL deaktivieren |

## Nur Watcher-Image bauen

```bash
docker build -t meipi-index-watcher .
docker run --rm \
  --network postgresql_default \
  -e IND_PG_HOST=pg-db \
  -e IND_PG_PASSWD=secret \
  -e IND_DOCROOT=/data \
  -e IND_TIKA_NOOCR_URL=http://tika:9998 \
  -e IND_WATCH_POOL_ID=1 \
  -v /pfad/zu/dateien:/data:ro \
  meipi-index-watcher
```

## Hinweise

- Apache Tika läuft mit `docker/tika-config.xml` (OCR-Parser deaktiviert).
- Einzelbild-Thumbnails laufen per PIL (kein CUDA/DALI nötig).
- Der Keyring wird im Container nicht genutzt; setzen Sie `IND_PG_PASSWD` direkt.
