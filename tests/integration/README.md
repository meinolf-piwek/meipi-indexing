# PostgreSQL integration tests

These tests use a **real** database. They call `recreate_tables()` and **drop all application tables** in the target database.

## Setup

1. Create an empty database (example name: `meipi_indexing_test`).
2. Ensure [pgvector](https://github.com/pgvector/pgvector) can be installed (`CREATE EXTENSION vector`).
3. Export the SQLAlchemy URL:

```bash
export MEIPI_TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/meipi_indexing_test"
```

## Run

```bash
# unit tests only (default)
pytest

# integration tests only
pytest -m integration

# everything
pytest -m ""
```

## Docker one-liner (optional)

```bash
docker run --rm -d --name meipi-pg-test -e POSTGRES_PASSWORD=postgres -p 55432:5432 pgvector/pgvector:pg16
export MEIPI_TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:55432/postgres"
pytest -m integration
docker stop meipi-pg-test
```
