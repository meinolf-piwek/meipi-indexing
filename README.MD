# Meipi Indexing

!!! Work in progress !!!
Dieses Projekt dient der Indexierung und Verwaltung von von Dateien. Aktuell ausgeprägt für Bilddateien.
Es erzeugt Tabellen mit Metadaten, Bild- und Textdaten in einer PostgreSQL Datenbank.
Als weitere Ausbaustufe werden mit einem KI-Modell Einbettungen erzeugt und als Vektoren in der Datenbank
gespeichert.
Das Coding ist in Python3, mit uv als Paketmanager.

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

Eine Dokumentation der API findet sich in [API](./doc/singlehtml/index.html)

Vorgehen:

    1. Konfiguration mit Hilfe von config.env
    2. Anlegen der Datenbankstruktur mit db_init (to_do)
    3. Einlesen der Metadaten der relevanten Dateien
    4. Erzeugen der Dateitypspezifischen DB-Einträge
    5. Generieren und Speichern der Embeddings
    6. Query-Oberfläche für Suchen



```bash

```

## Konfiguration

Bearbeite die Datei `config.json`, um die Einstellungen anzupassen.

## Mitwirken

Issues sind willkommen!

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
