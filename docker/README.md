# Docker: Index-Watcher

Dieser Stack startet Apache Tika und den Datei-Watcher. PostgreSQL läuft
extern (Container `pg-db` im Docker-Netzwerk `postgresql_default`).

Beim Start prüft der Watcher, ob Datenbank und Dateisystem übereinstimmen, und
meldet Schema-Informationen sowie Abweichungen (ohne sie zu beheben). Bei
Abweichungen beendet sich der Watcher mit Exit-Code 1 und startet nicht.
Derselbe Check ist separat verfügbar: ``meipi-index check-sync --pool-id 1 .``
(Exit-Code 1 bei Abweichungen). Tabellen, Datapool und Erstindexierung müssen
vorher angelegt werden.

## Voraussetzungen

- PostgreSQL mit pgvector erreichbar als `pg-db` im Netzwerk `postgresql_default`
- Netzwerk muss existieren: `docker network ls | grep postgresql_default`

```bash
export IND_DOCROOT=/pfad/zu/dateien   # oder: meipi-index --docroot /pfad/zu/dateien …
meipi-index create-tables
meipi-index create-pool --name default
meipi-index read-files --pool-id 1 .
```

Im Container ist `IND_DOCROOT` fest `/data`; der Host-Pfad kommt über `HOST_DATA_DIR`
(bzw. `HOST_DATA_DIR_2` für `watcher-2`).

## Schnellstart

```bash
cp docker/.env.example docker/.env
# IND_PG_*, IND_WATCH_*, HOST_DATA_DIR anpassen

docker compose --env-file docker/.env up --build -d
docker compose logs -f watcher-1 watcher-2
```

Stoppen:

```bash
docker compose down
```

## Volumes

| Host | Container | Zweck |
|------|-----------|-------|
| `HOST_DATA_DIR` (Standard: `./data`) | `/data` | Dateien für `watcher-1` |
| `HOST_DATA_DIR_2` (Standard: wie oben) | `/data` | Dateien für `watcher-2` |

## Wichtige Umgebungsvariablen

| Variable | Standard | Beschreibung |
|----------|----------|--------------|
| `IND_PG_HOST` | `pg-db` | PostgreSQL-Hostname |
| `IND_PG_USER` / `IND_PG_PASSWD` / `IND_PG_DATABASE` | — | DB-Zugangsdaten |
| `IND_WATCH_POOL_ID` | `1` | Datapool für Service `watcher-1` |
| `IND_WATCH_PATH` | `.` | Pfad unter `/data` (relativ zum Container-docroot) |
| `HOST_DATA_DIR` | `./data` | Host-Pfad für Pool 1, gemountet auf `/data` |
| `IND_WATCH_POOL_ID_2` | `2` | Datapool für Service `watcher-2` |
| `IND_WATCH_PATH_2` | `.` | Watch-Pfad für Pool 2 |
| `HOST_DATA_DIR_2` | wie `HOST_DATA_DIR` | Optional separater Host-Mount für Pool 2 |
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
- Watcher mit `restart: unless-stopped` starten nach Out-of-Sync-Exit automatisch
  neu; Abweichungen zuerst beheben oder `--no-startup-check` nur bewusst nutzen.
