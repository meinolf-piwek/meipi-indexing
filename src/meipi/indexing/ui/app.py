"""Streamlit UI for PostgreSQL full-text document search."""

from __future__ import annotations

import os
from datetime import datetime, time

import streamlit as st
from sqlalchemy import select

from meipi.indexing import appconf
from meipi.indexing.model import DBMeta, DBPool
from meipi.indexing.operations import DBOperations
from meipi.indexing.search import DocSearchHit, DocSearchResult, QueryMode, SortField, search_documents

LANG_OPTIONS = {
    "German": "german",
    "English": "english",
    "Simple (no stemming)": "simple",
}

MODE_OPTIONS: dict[str, QueryMode] = {
    "Web search (quotes, minus)": "websearch",
    "Plain (all words)": "plain",
    "Exact phrase": "phrase",
}

SORT_OPTIONS: dict[str, SortField] = {
    "Date": "sort_date",
    "Path": "path",
}

@st.cache_resource
def _db_for_pool(pool_id: int) -> DBOperations:
    return DBOperations(pool_id=pool_id)


@st.cache_data(ttl=30)
def _list_pools() -> list[dict]:
    ops = DBOperations(allow_no_pool=True)
    with ops.Session() as session:
        pools = session.scalars(select(DBPool).order_by(DBPool.pool)).all()
        return [
            {
                "id": pool.id,
                "name": pool.pool,
                "description": pool.description or "",
            }
            for pool in pools
        ]


@st.cache_data(ttl=30)
def _list_suffixes(pool_id: int) -> list[str]:
    ops = _db_for_pool(pool_id)
    with ops.Session() as session:
        return list(
            session.scalars(
                select(DBMeta.suffix)
                .where(DBMeta.pool_id == pool_id)
                .distinct()
                .order_by(DBMeta.suffix)
            )
        )


def _run_search(
    pool_id: int|None,
    query: str,
    lang: str,
    mode: QueryMode,
    limit: int,
    sort_by: SortField,
    sort_desc: bool,
    sort_date_from: datetime | None,
    sort_date_to: datetime | None,
    suffixes: list[str],
    path_prefix: str | None,
    include_metadata: bool,
) -> DocSearchResult:
    if pool_id is None:
        raise ValueError("pool_id is required")
    ops = _db_for_pool(pool_id)
    with ops.Session() as session:
        return search_documents(
            session,
            pool_id=pool_id,
            query=query,
            lang=lang,
            limit=limit,
            mode=mode,
            sort_by=sort_by,
            sort_desc=sort_desc,
            sort_date_from=sort_date_from,
            sort_date_to=sort_date_to,
            suffixes=suffixes or None,
            path_prefix=path_prefix,
            include_metadata=include_metadata,
        )


def _render_hit(hit: DocSearchHit, rootpath: str) -> None:
    full_path = os.path.join(rootpath, hit.path) if rootpath else hit.path
    with st.container(border=True):
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**{hit.fname}** `{hit.suffix}`")
            st.caption(full_path)
        with cols[1]:
            st.metric("Date", hit.sort_date.strftime("%Y-%m-%d %H:%M"))
        if hit.snippet.strip():
            st.markdown(hit.snippet, unsafe_allow_html=True)
        else:
            st.info("No snippet available for this match.")


def main() -> None:
    st.set_page_config(
        page_title="Meipi Document Search",
        page_icon="🔍",
        layout="wide",
    )
    st.title("Document full-text search")
    st.caption(
        "PostgreSQL full-text search over document content and file metadata "
        "(Tika fields, filename, path, content type)."
    )

    with st.sidebar:
        st.header("Connection")
        st.text_input(
            "Config env file (loaded at startup)",
            value=appconf.config_path,
            disabled=True,
            help=(
                "Set MEIPI_CONFIG_ENV before starting Streamlit to use another file. "
                "Restart the app after changing configuration."
            ),
        )
        st.text_input(
            "Document root (IND_DOCROOT)",
            value=appconf.resolved_docroot(),
            disabled=True,
            help="Filesystem root for indexed file paths. Set IND_DOCROOT before starting the app.",
        )
        if st.button("Reload pools", use_container_width=True):
            _list_pools.clear()
            _list_suffixes.clear()
            _db_for_pool.clear()

        try:
            pools = _list_pools()
        except Exception as exc:
            st.error(f"Database connection failed: {exc}")
            st.stop()

        if not pools:
            st.warning("No data pools found. Create one before searching.")
            st.stop()

        pool_labels = {
            p["id"]: f"{p['name']} (id {p['id']})" for p in pools
        }
        pool_id = st.selectbox(
            "Data pool",
            options=list(pool_labels.keys()),
            format_func=lambda pid: pool_labels[pid],
        )
        selected_pool = next(p for p in pools if p["id"] == pool_id)
        if selected_pool["description"]:
            st.caption(selected_pool["description"])

        st.divider()
        st.header("Search options")
        lang_label = st.selectbox("Language", options=list(LANG_OPTIONS.keys()))
        mode_label = st.selectbox("Query mode", options=list(MODE_OPTIONS.keys()))
        include_metadata = st.toggle(
            "Include metadata in search",
            value=True,
            help="Also match filename, path, content type, and Tika metadata fields.",
        )
        sort_label = st.selectbox("Sort by", options=list(SORT_OPTIONS.keys()))
        sort_desc = st.toggle("Descending", value=True)
        limit = st.number_input(
            "Max results shown",
            min_value=1,
            value=25,
            step=5,
            help="All matching documents are counted; only this many are displayed.",
        )

        st.divider()
        st.header("Filters")
        use_date_filter = st.toggle("Filter by date range", value=False)
        sort_date_from = None
        sort_date_to = None
        if use_date_filter:
            date_cols = st.columns(2)
            with date_cols[0]:
                from_date = st.date_input("From", value=None)
            with date_cols[1]:
                to_date = st.date_input("To", value=None)
            if from_date is not None:
                sort_date_from = datetime.combine(from_date, time.min)
            if to_date is not None:
                sort_date_to = datetime.combine(to_date, time.max)

        selected_suffixes = st.multiselect(
            "Suffix",
            options=_list_suffixes(pool_id),
            help="Leave empty to include all suffixes. Multiple values allowed.",
        )
        path_prefix = st.text_input(
            "Path starts with",
            placeholder="e.g. archive/2024/",
            help="Relative path prefix within the indexed document root.",
        )

        with st.expander("Query syntax help"):
            st.markdown(
                """
**Web search** (default): supports quoted phrases (`"annual report"`),
exclusion (`budget -draft`), and OR.

**Plain**: all words must appear (order ignored).

**Exact phrase**: words must appear adjacent and in order.

With metadata enabled, searches **extracted text** and **metadata**
(Tika JSON, filename, path).

Leave the query empty to list all documents matching the filters.

Examples: `Vertrag`, `"Projektplan"`, `Rechnung -Entwurf`, `image/tiff`
                """
            )

    query = st.text_input(
        "Search query",
        placeholder='e.g. Vertrag or "Projektplan"',
        label_visibility="collapsed",
    )
    search_clicked = st.button("Search", type="primary", use_container_width=False)

    if not search_clicked:
        st.info("Enter a query or set filters, then click Search.")
        return

    lang = LANG_OPTIONS[lang_label]
    mode = MODE_OPTIONS[mode_label]
    sort_by = SORT_OPTIONS[sort_label]
    with st.spinner("Searching…"):
        try:
            result = _run_search(
                pool_id,
                query,
                lang,
                mode,
                limit,
                sort_by,
                sort_desc,
                sort_date_from,
                sort_date_to,
                selected_suffixes,
                path_prefix,
                include_metadata,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    shown = len(result.hits)
    total = result.total_count
    if shown < total:
        st.subheader(f"Showing {shown} of {total} results")
    else:
        st.subheader(f"{total} result{'s' if total != 1 else ''}")
    if total == 0:
        st.warning("No documents matched your search or filters.")
        return

    for hit in result.hits:
        _render_hit(hit, appconf.resolved_docroot())


def launch() -> None:
    """Start the Streamlit server (console script entry point)."""
    import subprocess
    import sys

    from pathlib import Path

    app_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
