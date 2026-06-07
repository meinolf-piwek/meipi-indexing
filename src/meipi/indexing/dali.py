"""DALI-based image loading and resizing helpers with PIL fallback.

The module provides a high-throughput resize pipeline for generating thumbnails
from file paths plus picture ids. Failed DALI batches can be retried with
smaller batches and optionally with PIL-backed loading via DALI external source.
For single-image thumbnails without CUDA, use :mod:`meipi.indexing.thumbnail`.
"""
__all__ = ["PILLoader", "DALIImageResizer"]

from typing import List, Tuple, Sequence
from itertools import batched
import warnings
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener
warnings.filterwarnings("ignore", module="nvidia.dali.backend")
from nvidia import dali
from nvidia.dali.fn import resize, pad
from nvidia.dali.fn.readers import file as dali_file_reader
from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import cupy as cp
from .model import IdList
from .config import Config, resolve_config

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
        config: Config | None = None,
        #use_PIL: bool = False,
    ):
        config = resolve_config(config)
        self.config = config
        self.logger = config.logger
        self.files = files
        self.labels = labels
        if len(self.files) != len(self.labels):
            raise ValueError("Files and labels must have the same length")
        self.pipe_batch_size = pipe_batch_size
        self.num_threads = num_threads
        #self.use_PIL = use_PIL

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
        use_PIL: bool = False,
        # pkl_file: str=None,
        show_progress: bool = False,
    ) -> tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """Run one resize pass and return successes and failures.

        Returns:
            A tuple ``(images, labels, error_files, error_labels)`` where labels
            are ``pictures.id`` values.
        """
        if len(files) != len(labels):
            raise ValueError("Files and labels must have the same length")
        pipe_batch_size = min(batch_size, self.pipe_batch_size)
        if use_PIL:
            pipe = self.pipePIL(files, labels)
            reader_name = None
        else:
            reader_name = "Reader"
            pipe = self.pipedali(files, labels)
        err = []
        respics: List[np.ndarray] = []
        reslabels: List[int] = []
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
            self.logger.error(f"Caught Error in process(): {e}")
            #print(f"Caught Error: {e}")
        errlabels = [x[1] for x in err if x[1] not in reslabels]
        errfiles = [x[0] for x in err if x[1] in errlabels]
        return respics, reslabels, errfiles, errlabels

    
    def resize_pics(
        self, piclist: IdList, batch_size: int, use_PIL: bool
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """Create thumbnails for all pictures in ``piclist``.

        Der DALI-Image-Resizer ist wesentlich performanter als der PIL-Image-Resizer.
        Außerdem sind größere Batches deutlich performanter als kleinere Batches.
        Schlägt allerdings eine Operation fehl, so wird der gesamte Batch abgebrochen
        und es wird mit dem nächsten Batch fortgefahren.
        Daher wird zuerst mit DALI und einer großen Batchgröße versucht, die Bilder zu laden.
        Die fehlerhaften Bilder werden dann mit Batchgröße 1 verarbeitet.
        Die restlichen Fehler werden dann mit PIL verarbeitet.
        Das Ergebnis ist eine Liste von Bildern, Labels, Fehlerdateipfaden und Fehlerlabels.
        !Achtung: Die Reihenfolge der Bilder in den Listen ist nicht die gleiche wie
        in der Eingabeliste.

        Args:
            piclist (IdList): (Dateipfad, ``filemeta.id``)
            batch_size (int): Anzahl der Bilder, die in einem Batch verarbeitet werden sollen
            use_PIL (bool): Ob die Thumbnails mit PIL erstellt werden sollen (True) oder mit DALI (False)

        Returns:
            ``(thumbnails, meta_ids, failed_paths, failed_meta_ids)``.
        """
        # Durchgang 1: DALI und große Batchgröße
        batches = batched(piclist, batch_size)
        grespics, gresids, gerrfiles, gerrids = ([], [], [], [])
        for batch in tqdm(
            batches, total=(len(piclist) // batch_size), desc="DALI and large batch size"
        ):
            files, pic_ids = zip(*batch)
            respics, resids, errfiles, errids = self.process(
                files=files, labels=pic_ids, batch_size=batch_size, show_progress=True, use_PIL=use_PIL
            )
            grespics.extend(respics)
            gresids.extend(resids)
            gerrfiles.extend(errfiles)
            gerrids.extend(errids)
        self.logger.info(f"DALI and large batch size: {len(grespics)} Bilder erfolgreich verarbeitet, {len(gerrfiles)} Bilder fehlgeschlagen")
        print(f"DALI and large batch size: {len(grespics)} Bilder erfolgreich verarbeitet, {len(gerrfiles)} Bilder fehlgeschlagen")
        # Wenn keine Bilder fehlgeschlagen sind, gib die Ergebnisse zurück
        if len(gerrfiles) == 0:
            return grespics, gresids, [], []  # alle Bilder erfolgreich verarbeitet
        
        # Durchgang 2: DALI und kleine Batchgröße
        batches = batched(zip(gerrfiles, gerrids), 1)
        gerrfiles2, gerrids2 = ([], [])
        for batch in tqdm(
            batches, total=(len(gerrfiles)), desc="DALI and small batch size"
        ):
            files, pic_ids = zip(*batch)
            respics, resids, errfiles, errids = self.process(
                files=files, labels=pic_ids, batch_size=1, show_progress=True, use_PIL=False
            )
            grespics.extend(respics)
            gresids.extend(resids)
            gerrfiles2.extend(errfiles)
            gerrids2.extend(errids)
        self.logger.info(f"DALI and small batch size: {len(grespics)} Bilder erfolgreich verarbeitet, {len(gerrfiles2)} Bilder fehlgeschlagen")
        print(f"DALI and small batch size: {len(grespics)} Bilder erfolgreich verarbeitet, {len(gerrfiles2)} Bilder fehlgeschlagen")
        # Wenn keine Bilder fehlgeschlagen sind, gib die Ergebnisse zurück
        if len(gerrfiles2) == 0:
            return grespics, gresids, [], []
        
        # Durchgang 3: PIL und kleine Batchgröße
        batches = batched(zip(gerrfiles2, gerrids2), 1)
        gerrfiles3, gerrids3 = ([], [])
        for batch in tqdm(
            batches, total=(len(gerrfiles2) // 1), desc="PIL and small batch size"
        ):
            files, pic_ids = zip(*batch)
            respics, resids, errfiles, errids = self.process(
                files=files, labels=pic_ids, batch_size=1, show_progress=True, use_PIL=True
            )
            grespics.extend(respics)
            gresids.extend(resids)
            gerrfiles3.extend(errfiles)
            gerrids3.extend(errids)
        self.logger.info(f"PIL and small batch size: {len(grespics)} Bilder erfolgreich verarbeitet, {len(gerrfiles3)} Bilder fehlgeschlagen")
        return grespics, gresids, gerrfiles3, gerrids3


