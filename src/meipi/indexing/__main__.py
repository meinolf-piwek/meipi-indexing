from pathlib import Path
from itertools import batched
from pickle import dump
from tqdm import tqdm
from .preprocess import tika_parse
from . import appconf
#from .preprocess import get_DBMeta_from_file
#from .db import pgEngine, DBDoc, DBPic


#appconf = Config.from_env_file()
filelist = [root/file for root,_,files in Path(appconf.docroot+"/mobile").walk() for file in files ]

batches = batched(filelist, n=1000)
docs=[]
for batch in tqdm(batches):
    docs.extend(tika_parse(batch))
with open("pickle.pkl", "wb") as f:
    dump(docs,f)