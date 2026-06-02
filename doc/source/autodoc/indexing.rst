meipi.indexing package
======================

Diese Seite beschreibt die wichtigsten Module des Pakets
``meipi.indexing`` ohne Laufzeit-Importe waehrend des Sphinx-Builds.
Damit bleibt die Dokumentation auch in Umgebungen ohne volle Runtime-
Abhaengigkeiten stabil.

Kernmodule
----------

- ``meipi.indexing.config``: App-Konfiguration (ENV/Keyring/DB-URL)
- ``meipi.indexing.model``: SQLAlchemy-Modelle fuer Pools, File-Metadaten und Typ-Tabellen
- ``meipi.indexing.operations``: DB-Workflows und asynchrone Datei-Verarbeitung
- ``meipi.indexing.search``: PostgreSQL-Volltextsuche
- ``meipi.indexing.picture``: Thumbnail-Generierung und Bild-Resize-Pipeline
- ``meipi.indexing.embedding``: Batching und Erzeugung von Bild-Embeddings
- ``meipi.indexing.srcdocs``: experimentelle Speicherung binaerer Bilddaten
- ``meipi.indexing.langchain``: veraltete Integrationsreste

Hinweis zum API-Build
---------------------

Fuer echte autodoc-Ausgaben mit importierten Symbolen sollte der Build in der
Projektumgebung mit allen Abhaengigkeiten laufen, z.B. mit ``uv run``.
