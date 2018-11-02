
import pandas as pd
import gzip
import io

import logging
import time
import json
import tqdm



from gensim.models import FastText, Word2Vec
from gensim.test.utils import get_tmpfile

#____ read Raw data to pandas

def parse(path):
  g = gzip.open(path, 'rb')
  f = io.BufferedReader(g)  
  for l in g:
    yield eval(l)

def getDF(path, num = 1000):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
    if i > num:
        break
    if i % 10000 == 0:
      print(i)
  return pd.DataFrame.from_dict(df, orient='index')


def get_docs(num):
  logging.info("Loading data...")
  df = getDF('metadata.json.gz', num)
  docs = df.related.map(lambda x: x.get("also_bought") if type(x)==dict else None).dropna()
  logging.info("Parsing data...")
  return docs[docs.map(lambda x: len(x) > 2)].reset_index(drop=True).tolist()

def build_maping(df):
  maps =  df[["asin", "title"]].to_dict(orient = "index")
  list_of_dicts = {v["asin"]: v["title"] for k, v in maps.items()}
  return list_of_dicts

def dump_mappings(mappings):
  logging.info("Dumping mappings")
  json.dump(mappings, open("mappings_image.json", "w") )

def load_mappings():
  return json.load(open("mappings.json", "r"))

def dump_model(model, name):
  fname = get_tmpfile(name)
  logging.info("saving model...")
  model.save(fname)

def main():
  logging.basicConfig(level = logging.INFO)
  t1 = time.time()
  docs = get_docs(1e25)
  model = Word2Vec(docs)
  logging.info("saving model...")
  model.save("w2v.model.all")
  #dump_model(model, "fasttext.model")
  t2 = time.time()
  print(t2 - t1)

def mapping_main():
  logging.basicConfig(level = logging.INFO)
  df = getDF('metadata.json.gz', 1e25)
  mappings = build_maping(df)
  dump_mappings(mappings)


if __name__ == '__main__':
  main()


  #dump_mappings(df)
