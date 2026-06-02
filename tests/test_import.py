"""Smoke tests for package imports."""


def test_public_imports():
    from meipi.indexing import (
        AsyncFileOperations,
        Base,
        Config,
        DBDinoV2Vector,
        DBDoc,
        DBMeta,
        DBOperations,
        DBPic,
        DBPool,
        appconf,
    )

    assert Config is not None
    assert appconf is not None
    assert DBOperations.__name__ == "DBOperations"


def test_id_list_alias():
    from meipi.indexing.model import IdList

    sample: IdList = [("/tmp/a.jpg", 1)]
    assert sample[0][1] == 1
