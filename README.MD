# Meipi Indexing

Dieses Projekt dient der Indexierung und Verwaltung von von Dateien. Aktuell ausgeprägt für Bilddateien.
Es erzeugt Tabellen mit Metadaten, Bild- und Textdaten in einer PostgreSQL Datenbank

## Features

- Metadatenextraktion mit Apache tika
- Für Bilddateien dann XMP-Extraktion, Thumb-Generierung und pHash-Generierung
- für Textdateien Volltextsuche in Postgresql
- to do: Vektoreinbettung für Dokument- und Bildsuche


## Installation

```bash
git clone https://github.com/meinolf-piwek/meipi-indexing.git
cd meipi-indexing
uv pip install .
```

## Nutzung

```bash

```

## Konfiguration

Bearbeite die Datei `config.json`, um die Einstellungen anzupassen.

## Mitwirken

Pull Requests und Issues sind willkommen!

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.