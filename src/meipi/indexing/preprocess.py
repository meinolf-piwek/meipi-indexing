"""Load and preprocess documents"""
#pylint: disable=W0718
import os
from datetime import datetime
import json
from hashlib import file_digest
from itertools import batched
from tqdm.auto import tqdm
import tika.parser as tp
import langchain_core.documents as lcd
from libxmp.utils import file_to_dict
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener
from nvidia import dali
from nvidia.dali.fn import resize, pad
from nvidia.dali.fn.readers import file as dali_file_reader
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import cupy as cp
from . import appconf
from .db import DBDoc, DBPic, DBMeta

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_DBMeta_from_file(file:str, docroot:str, pool:str)->DBMeta:
    """
    Get DBMeta object from file path.
    """
    meta = tika_get_meta(file)
    meta = json.loads(json.dumps(meta).replace("\\u0000","null"))
    path = file.replace(docroot, "")
    os_fdate:str = meta.get("fdate", datetime.now().isoformat())
    doc_cdate:str = meta.get("dcterms:created", None)
    content_length = meta.get("Content-Length", 0)
    if isinstance(content_length, list):
        content_length:int = content_length[0]
    if isinstance(doc_cdate, list):
        doc_cdate:str = doc_cdate[0]
    with open(file, "rb") as f:
        sha256 = file_digest(f, "sha256")
    try:
        dbmeta = DBMeta(
            id=None,
            pool=pool,
            path=path,
            fname=os.path.basename(path),
            fdate=os_fdate,
            sort_date=os_fdate if doc_cdate is None else doc_cdate,
            fsize=meta.get("fsize", 0),
            clength=content_length,
            ctype=meta.get("Content-Type", "unk/unk"),
            md_keys=list(meta.keys()),
            suffix=os.path.splitext(path)[1],
            sha256=sha256.digest(),
            meta_data=meta,
            )
    except Exception as e:
        appconf.logger.error("Error %s creating DBMeta fpath %s", e, file)
        return None
    else:
        return dbmeta

def tika_get_meta(file:str)->dict:
    """
    Get metadata of a file using Tika.
    """
    headers = {"X-Tika-OCRskipOcr": "true"}
    try:
        parsed = tp.from_file(file, xmlContent=False, requestOptions={"headers": headers})
        parsed["metadata"]["file"] = file
        parsed["metadata"]["fdate"] = datetime.fromtimestamp(os.path.getmtime(file)).isoformat()
        parsed["metadata"]["fsize"] = os.path.getsize(file)
    except Exception as e:
        appconf.logger.error("Error %s parsing file %s", e, file)
        return None
    return parsed["metadata"]

def tika_parse(files:list[str])->list[lcd.Document]:
    """
    Parse a file using Tika and return the parsed content.
    """
    headers = {"X-Tika-OCRskipOcr": "true"}
    doclist = []
    for file in files:
        try:
            parsed = tp.from_file(file, xmlContent=False, requestOptions={"headers": headers})
            parsed["metadata"]["file"] = file
            parsed["metadata"]["fdate"] = datetime.fromtimestamp(os.path.getmtime(file)).isoformat()
            parsed["metadata"]["fsize"] = os.path.getsize(file)
            page_content = parsed.get("content","")
            page_content = "" if not page_content else page_content
            doclist.append(lcd.Document(page_content=page_content,metadata=parsed["metadata"]))
        except Exception as e:
            appconf.logger.error("Error %s parsing file %s", e, file)
            return None
    return doclist

def dbdoc_from_lcdoc(lcdoc: lcd.Document, docroot) -> DBDoc:
    """Erzeuge DBDocumente aus LangChain Document"""
    meta: dict = lcdoc.metadata
    os_fdate:str = meta.get("fdate", datetime.now().isoformat())
    doc_cdate:str = meta.get("dcterms:created", "1990-01-01T00:00:00")
    if isinstance(doc_cdate, list):
        doc_cdate:str = doc_cdate[0]
    fpath:str = meta.get("file", "")
    content_length = meta.get("Content-Length", 0)
    if isinstance(content_length, list):
        content_length:int = content_length[0]
    try:
        dbdoc = DBDoc(
        id=os.path.relpath(fpath, start=docroot),
        fname=os.path.basename(fpath),
        suffix=os.path.splitext(fpath)[1],
        fdate=os_fdate,
        sort_date=os_fdate if doc_cdate is None else doc_cdate,
        fsize=meta.get("fsize", 0),
        clength=content_length,
        ctype=meta.get("Content-Type", "unk/unk"),
        md_keys=list(meta.keys()),
        inhalt=lcdoc.page_content if lcdoc.page_content else "",
        meta_data=meta,
        )
    except Exception as e:
        appconf.logger.error("Error %s creating DBDoc fpath %s", e, fpath)
        return None
    else:
        return dbdoc

def dbpic_from_dbmeta(dbmeta: DBMeta,docroot:str)->DBPic:
    """
    Get DBPic object from DBMeta object.
    """
    try:
        xmp = file_to_dict(docroot+dbmeta.path)
        xmpdict = dict([(a,b.replace("\\u0000","null")) for data in xmp.values() for a, b, _ in data])
        dbpic = DBPic(
            **dbmeta.as_dict(),
            xmp=xmpdict,
            )
    except Exception as e:
        appconf.logger.error("Error %s creating DBPic fpath %s", e, dbmeta.path)
        return None
    else:
        return dbpic
    


class PILLoader(object):
    def __init__(self, files:list[str], labels: list[str], batch_size):
        assert len(files) == len(labels), "Length of files and labels do not match"
        self.batch_size = batch_size
        self.files = files
        self.labels = labels
        
        
    def __iter__(self):
        self.batches = batched(zip(self.files,self.labels), self.batch_size)
        return self
    
    def __next__(self):
        batch = next(self.batches)
        outfiles = [cp.asarray(Image.open(file),dtype=cp.uint8) for file, _ in batch]
        outlabels = [cp.array([label], dtype= cp.int64) for _, label in batch]
        return (outfiles, outlabels)

class DALIImageResizer:
    """
    DALI Image Resizer
    """
    
    def __init__(self, 
            files:list[str]=[], labels:list[int]=[], pipe_batch_size:int=1, num_threads:int=1, use_PIL: bool = False):
        if len(files) != len(labels):
            raise ValueError("Files and labels must have the same length")
        self.files = files
        self.labels = labels
        self.pipe_batch_size = pipe_batch_size
        self.num_threads = num_threads
        self.use_PIL = use_PIL
    
        
    def pipedali(self,batch_files, batch_labels):
        @dali.pipeline_def(batch_size=self.pipe_batch_size, num_threads=self.num_threads, enable_conditionals=False)
        def pipe():
            inp, label = dali_file_reader(  #pylint: disable=unpacking-non-sequence
                files=batch_files, 
                labels=batch_labels, 
                random_shuffle=False, 
                name="Reader"
                ) 
            decoded = dali.fn.decoders.image(inp, device="mixed", output_type=dali.types.DALIImageType.RGB)
            resized = resize(decoded, resize_longer=224)
            padded = pad(resized,axes=(0,1),shape=(224,224))
            return padded, label
        return pipe
    
    def pipePIL(self, batch_files, batch_labels):
        extiter = PILLoader(files=batch_files, labels=batch_labels, batch_size=self.pipe_batch_size)
        @dali.pipeline_def(batch_size=self.pipe_batch_size, num_threads=self.num_threads, enable_conditionals=False)
        def pipe():
            decoded, label = dali.fn.external_source(source = extiter, num_outputs=2)
            resized = resize(decoded, resize_longer=224)
            padded = pad(resized,axes=(0,1),shape=(224,224))
            return padded, label
        return pipe

            
    
    def process(self, 
                files:list[str]=None, 
                labels:list[int]=None, 
                batch_size:int=1,
                #pkl_file: str=None,
                show_progress:bool=False,                
                )-> tuple[list, list,list,list]:
        """Verabeitet die Pipeline in Batches und gibt die Ergebnisse zurück.
        Rückgabe: (Bilder, Labels, Fehlerdateipfade, Fehlerlabels)"""
        if len(files) != len(labels):
            raise ValueError("Files and labels must have the same length")
        pipe_batch_size = min(batch_size, self.pipe_batch_size)
        if self.use_PIL:
            pipe = self.pipePIL(files, labels)
            reader_name = None
        else:
            reader_name = "Reader"
            pipe = self.pipedali(files,labels)
        err = []
        respics = []
        reslabels = []
        try:
            dali_iter = DALIClassificationIterator(
                pipe(batch_size=pipe_batch_size), 
                reader_name=reader_name,
                last_batch_policy=LastBatchPolicy.PARTIAL)
            if show_progress:
                dali_iter = tqdm(dali_iter, total=len(files)//pipe_batch_size+1)
            for result in dali_iter:
                for r in result:                        
                    respics.extend(r["data"].cpu().numpy())
                    reslabels.extend(r["label"].flatten().tolist())
        except Exception as e:
            err.extend(zip(files, labels))
            print(f"Caught Error: {e}")
        errlabels = [x[1] for x in err if x[1] not in reslabels]
        errfiles = [x[0] for x in err if x[1] in errlabels]
        return respics, reslabels, errfiles, errlabels

    def process_batched(self, 
                files:list[str]=None, 
                labels:list[int]=None, 
                batch_size:int=1,
                show_progress:bool=False
                )-> tuple[list, list,list,list]:
        """Verabeitet die Pipeline in Batches und gibt die Ergebnisse zurück.
        Rückgabe: (Bilder, Labels, Fehlerdateipfade, Fehlerlabels)"""
        if files is None:
            files = self.files
        if labels is None:
            labels = self.labels
        if len(files) != len(labels):
            raise ValueError("Files and labels must have the same length")
        pipe_batch_size = min(batch_size, self.pipe_batch_size)
        err = []
        respics = []
        reslabels = []
        batches = batched(zip(files, labels), batch_size)
        batchiterator = enumerate(batches)
        if show_progress:
            batchiterator = enumerate(tqdm(batches, total=len(files)//batch_size+1))
        for bnum, batch in batchiterator:
            fl = [el[0] for el in batch]
            la = [el[1] for el in batch]
            err = []
            respics = []
            reslabels = []
            if self.use_PIL:
                pipe = self.pipePIL(fl, la)
            else:
                pipe = self.pipedali(fl,la)
            try:
                dali_iter = DALIClassificationIterator(
                    pipelines=pipe(batch_size=pipe_batch_size),
                    reader_name="Reader",
                    last_batch_policy=LastBatchPolicy.PARTIAL)
                for result in dali_iter:
                    for r in result:                        
                        respics.extend(r["data"].cpu().numpy())
                        reslabels.extend(r["label"].flatten().tolist())
            except Exception as e:
                err.extend(zip(fl, la))
                print(f"Caught Error processing batch {bnum}: {e}")
        errlabels = [x[1] for x in err if x[1] not in reslabels]
        errfiles = [x[0] for x in err if x[1] in errlabels]
        return respics, reslabels, errfiles, errlabels
