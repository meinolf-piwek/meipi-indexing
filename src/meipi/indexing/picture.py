""" Modul zum Laden und Vorverarbeiten von Bildern mit DALI.
    Es können Bilder mit DALI oder PIL geladen werden.
    Die Bilder werden auf eine Größe von 224x224 skaliert und gepaddet.
    Die Funktion process() verarbeitet die Bilder in Batches und gibt die Ergebnisse zurück.
    Eingabe: Liste von Dateipfaden und Labels, Batchgröße in der Pipeline, Zahl der Threads.
    Ausgabe: Tupel aus vier Listen: (Bilder, Labels, Fehlerdateipfade, Fehlerlabels)
    
"""
__all__ = ["PILLoader", "DALIImageResizer", "resize_pics"]

from typing import List, Tuple, Sequence
from itertools import batched
from tqdm.auto import tqdm
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener
from nvidia import dali
from nvidia.dali.fn import resize, pad
from nvidia.dali.fn.readers import file as dali_file_reader
from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import cupy as cp
from .model import IdList

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PILLoader(object):
    """PIL Loader for DALI External Source.
    Lädt Bilder mit PIL und gibt sie als CuPy-Arrays zurück.
    Genauer: Es wird ein Tupel aus zwei Listen der Länge "batch-size" zurückgegeben:
    Eine Liste von Bildern als CuPy-Arrays und eine Liste von Labels als CuPy-Arrays.
    """

    def __init__(self, files: Sequence[str], labels: Sequence[str], batch_size):
        assert len(files) == len(labels), "Length of files and labels do not match"
        self.batch_size = batch_size
        self.files = files
        self.labels = labels
        self.batches = batched(zip(files, labels), batch_size)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[list, list]:
        batch = next(self.batches)
        outfiles = [cp.asarray(Image.open(file), dtype=cp.uint8) for file, _ in batch]
        outlabels = [cp.array([label], dtype=cp.int64) for _, label in batch]
        return (outfiles, outlabels)


class DALIImageResizer:
    """
    DALI Image Resizer
    Klasse zum Laden und Vorverarbeiten von Bildern mit DALI.
    Es können Bilder mit DALI oder PIL geladen werden.
    Die Bilder werden auf eine Größe von 224x224 skaliert und gepaddet.
    Die Funktion process() verarbeitet die Bilder in Batches und gibt die Ergebnisse zurück.
    Eingabe: Liste von Dateipfaden und Labels, Batchgröße in der Pipeline, Zahl der Threads.
    Ausgabe: Tupel aus vier Listen: (Bilder, Labels, Fehlerdateipfade, Fehlerlabels)
    """
    def __init__(
        self,
        files: Sequence[str] = (),
        labels: Sequence[int] = (),
        pipe_batch_size: int = 1,
        num_threads: int = 1,
        use_PIL: bool = False,
    ):
        self.files = files
        self.labels = labels
        if len(self.files) != len(self.labels):
            raise ValueError("Files and labels must have the same length")
        self.pipe_batch_size = pipe_batch_size
        self.num_threads = num_threads
        self.use_PIL = use_PIL

    def pipedali(self, batch_files, batch_labels):
        """Erstellt eine DALI-Pipeline zum Laden und Vorverarbeiten von Bildern.
        Die Pipeline liest die Bilder mit dem DALI-File-Reader, dekodiert sie,
        skaliert sie auf eine Größe von 224x224 und paddet sie.
        Die Funktion gibt die Pipeline zurück, die in der Funktion process() verwendet wird.
        """

        @dali.pipeline_def(
            batch_size=self.pipe_batch_size,
            num_threads=self.num_threads,
            enable_conditionals=False,
        )
        def pipe():
            inp, label = dali_file_reader(  # pylint: disable=unpacking-non-sequence
                files=batch_files,
                labels=batch_labels,
                random_shuffle=False,
                name="Reader",
            )
            decoded =dali.fn.decoders.image(
                inp, device="mixed", output_type=dali.types.DALIImageType.RGB
            )
            resized = resize(decoded, resize_longer=224) 
            padded = pad(resized, axes=(0, 1), shape=(224, 224))  #type: ignore
            return padded, label

        return pipe

    def pipePIL(self, batch_files, batch_labels):
        """Erstellt eine DALI-Pipeline zum Laden und Vorverarbeiten von Bildern mit PIL.
        Die Pipeline liest die Bilder mit einem externen Iterator, dekodiert sie,
        skaliert sie auf eine Größe von 224x224 und paddet sie.
        Die Funktion gibt die Pipeline zurück, die in der Funktion process() verwendet wird.
        Wie "pipedali", aber mit einem externen Iterator, 
        der die Bilder mit PIL lädt und als CuPy-Arrays zurückgibt.
        """
        extiter = PILLoader(
            files=batch_files, labels=batch_labels, batch_size=self.pipe_batch_size
        )

        @dali.pipeline_def(
            batch_size=self.pipe_batch_size,
            num_threads=self.num_threads,
            enable_conditionals=False,
        )
        def pipe():
            decoded, label = dali.fn.external_source(source=extiter, num_outputs=2)
            resized = resize(decoded, resize_longer=224)
            padded = pad(resized, axes=(0, 1), shape=(224, 224)) #type: ignore
            return padded, label

        return pipe

    def process(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        batch_size: int = 1,
        # pkl_file: str=None,
        show_progress: bool = False,
    ) -> tuple[list, list, list, list]:
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
            pipe = self.pipedali(files, labels)
        err = []
        respics = []
        reslabels = []
        try:
            dali_iter = DALIClassificationIterator(
                pipe(batch_size=pipe_batch_size),
                reader_name=reader_name,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
            if show_progress:
                dali_iter = tqdm(dali_iter, total=len(files) // batch_size + 1)
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

    def process_batched(
        self,
        files: Sequence[str]|tuple[()] = (),
        labels: Sequence[int]|tuple[()] = (),
        batch_size: int = 1,
        show_progress: bool = False,
    ) -> tuple[list, list, list, list]:
        """Verabeitet die Pipeline in Batches und gibt die Ergebnisse zurück.
        Rückgabe: (Bilder, Labels, Fehlerdateipfade, Fehlerlabels)"""
        if files == ():
            files = self.files
        if labels == ():
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
            batchiterator = enumerate(tqdm(batches, total=len(files) // batch_size + 1))
        for bnum, batch in batchiterator:
            fl = [el[0] for el in batch]
            la = [el[1] for el in batch]
            err = []
            respics = []
            reslabels = []
            if self.use_PIL:
                pipe = self.pipePIL(fl, la)
            else:
                pipe = self.pipedali(fl, la)
            try:
                dali_iter = DALIClassificationIterator(
                    pipelines=pipe(batch_size=pipe_batch_size),
                    reader_name="Reader",
                    last_batch_policy=LastBatchPolicy.PARTIAL,
                )
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


def resize_pics(piclist: IdList, batch_size: int, pipe_batch_size: int, use_PIL: bool)-> Tuple[
    List[bytes], List[int], List[str], List[int]]:
    """Erstellt Thumbnails für die Bilder in piclist

    Args:
        piclist (IdList): Liste von Paaren aus Dateipfad und id
        batch_size (int): Anzahl der Bilder, die in einem Batch verarbeitet werden sollen
        pipe_batch_size (int): Anzahl Bilder pro Batch, die an die DALI-Pipeline übergeben werden sollen
        use_PIL (bool): Ob die Thumbnails mit PIL erstellt werden sollen (True) oder mit DALI (False)

    Returns:
        Tuple[List[bytes], List[int], List[str], List[int]]: Vier Listen: 1. Thumbnails als Byte-Arrays, 
        2. zugehörige ids, 3. Pfad der fehlgeschlagene Bilder, 4. ids der fehlgeschlagenen Bilder   
    """
    image_resizer = DALIImageResizer(
        pipe_batch_size=pipe_batch_size, num_threads=4, use_PIL=use_PIL
    )
    batches = batched(piclist, batch_size)
    grespics, greslabels, gerrfiles, gerrlabels = ([], [], [], [])
    for batch in tqdm(batches, total=(len(piclist) // batch_size)):
        files, labels= zip(*batch)
        respics, reslabels, errfiles, errlabels = image_resizer.process(
            files=files, labels=labels, batch_size=batch_size, show_progress=False
        )
        grespics.extend(respics)
        greslabels.extend(reslabels)
        gerrfiles.extend(errfiles)
        gerrlabels.extend(errlabels)
    return grespics, greslabels, gerrfiles, gerrlabels
