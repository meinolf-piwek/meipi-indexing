"""Deprecated LangChain integration stubs.

The active indexing pipeline no longer depends on this module. It remains in
the repository as historical reference until the replacement workflow is fully
stabilized.
"""


# import os
# from datetime import datetime
# import langchain_core.documents as lcd
# from .db.model import DBDoc
# from . import appconf

# def dbdoc_from_lcdoc(lcdoc: lcd.Document, docroot: str, pool: str) -> DBDoc|None:
#     """Erzeuge DBDocumente aus LangChain Document"""
#     meta: dict = lcdoc.metadata
#     os_fdate: str = meta.get("fdate", datetime.now().isoformat())
#     doc_cdate: str = meta.get("dcterms:created", "1990-01-01T00:00:00")
#     if isinstance(doc_cdate, list):
#         doc_cdate: str = doc_cdate[0]
#     fpath: str = meta.get("file", "")
#     suffix = os.path.splitext(fpath)[1]
#     content_length = meta.get("Content-Length", 0)
#     if isinstance(content_length, list):
#         content_length: int = content_length[0]
#     try:
#         dbdoc = DBDoc(
#             pool=pool,
#             path=os.path.relpath(fpath, start=docroot),
#             fname=os.path.basename(fpath),
#             suffix=suffix,
#             fdate=os_fdate,
#             sort_date=os_fdate if doc_cdate is None else doc_cdate,
#             fsize=meta.get("fsize", 0),
#             clength=content_length,
#             ctype=meta.get("Content-Type", "unk/unk"),
#             md_keys=list(meta.keys()),
#             ftype=appconf.get_ftype(suffix),
#             inhalt=lcdoc.page_content if lcdoc.page_content else "",
#             meta_data=meta,
#         )
#     except Exception as e:
#         appconf.logger.error("Error %s creating DBDoc fpath %s", e, fpath)
#         return None
#     else:
#         return dbdoc


# def tika_parse(files: Iterable[str]) -> list[lcd.Document]:
#     """
#     Parse a file using Tika and return the parsed content. file, fdate und fsize kommen vom os
#     """
#     headers = {"X-Tika-OCRskipOcr": "true"}
#     doclist = []
#     for file in files:
#         try:
#             parsed = tp.from_file(
#                 file, xmlContent=False, requestOptions={"headers": headers}
#             )
#             if not isinstance(parsed, dict):
#                 appconf.logger.error("Unexpected Tika response type %s for %s", type(parsed), file)
#                 return []
#             meta: dict = parsed.get("metadata", {}) # type: ignore
#             meta["file"] = file
#             meta["fdate"] = datetime.fromtimestamp(
#                 os.path.getmtime(file)
#             ).isoformat()
#             meta["fsize"] = os.path.getsize(file)
#             page_content = parsed.get("content", "")
#             page_content = "" if not page_content else page_content
#             doclist.append(
#                 lcd.Document(page_content=page_content, metadata=meta)
#             )
#         except Exception as e:
#             appconf.logger.error("Error %s parsing file %s", e, file)
#             return []
#     return doclist

