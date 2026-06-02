# meipi-indexing

`meipi-indexing` indexiert Dokumente, Bilder und Videos in einer PostgreSQL-Datenbank.
Gespeichert werden Dateimetadaten, extrahierter Text (Apache Tika), bildspezifische
Metadaten (XMP), Thumbnails und pHash-Werte. Auf dieser Basis sind Volltextsuche und
später auch Vektor-Workflows moeglich.

## Features

- Dateibasierte Indexierung ueber Datenpools (`datapools`)
- Metadaten- und Textextraktion mit Tika
- Volltextsuche in PostgreSQL (`tsvector` + `tsquery`)
- Bildverarbeitung mit Thumbnail-Generierung und pHash
- Alembic-Migrationen fuer das Datenbankschema

## Voraussetzungen

- Python 3.13+
- PostgreSQL mit `pgvector`-Erweiterung
- laufender Tika-Server (mit oder ohne OCR)
- optional CUDA/DALI fuer performante Bild-Preprocessing-Pipelines

## Installation

```bash
git clone https://github.com/meinolf-piwek/meipi-indexing.git
cd meipi-indexing
uv sync
```

Fuer eine klassische Paketinstallation:

```bash
uv pip install .
```

## Konfiguration

Die Konfiguration wird aus `config.env` (oder einer alternativen Datei via
`MEIPI_CONFIG_ENV`) geladen.

Wichtige Variablen:

- `IND_PG_HOST`, `IND_PG_PORT`, `IND_PG_USER`, `IND_PG_DATABASE`
- `IND_PG_API_KEY` (Keyring-Schluessel fuer das DB-Passwort)
- `IND_DATADIR`
- `IND_DOCSUF`, `IND_PICSUF`, `IND_VIDSUF`
- `IND_TIKA_NOOCR_URL`, `IND_TIKA_OCRURL`

Beispiel:

```bash
export MEIPI_CONFIG_ENV=./config.env
```

Hinweis: Die globale Konfiguration wird beim Import geladen. Nach Aenderungen an
Umgebungsvariablen oder `config.env` ist ein Prozess-Neustart noetig.

## Nutzung

CLI-Einstieg:

```bash
meipi-index --help
```

UI-Suche starten:

```bash
meipi-search
```

Tests ausfuehren:

```bash
pytest
```

## Dokumentation

- API-Dokumentation (Read the Docs):
  [https://meipi-indexing.readthedocs.io/de/latest/](https://meipi-indexing.readthedocs.io/de/latest/)
- Sphinx-Quellen liegen unter `doc/source`

### Docs Build (lokal)

Fuer einen sauberen Build mit importierbaren Projektmodulen sollte Sphinx in der
Projektumgebung ausgefuehrt werden:

```bash
uv sync
uv run --python 3.13 python -m sphinx -b html doc/source doc/build/html
```

Optional mit Warnungen als Fehler (fuer CI):

```bash
uv run --python 3.13 python -m sphinx -W -b html doc/source doc/build/html
```

## Mitwirken

Issues und Pull Requests sind willkommen.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
