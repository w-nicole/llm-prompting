import os, json, tqdm, re, string
import numpy as np
from tqdm import tqdm
from utils import *

from openai import OpenAI
client = OpenAI()

def remove_punc_and_lower(str):
    return re.sub('['+string.punctuation+']', '', str.lower())

def cosine_similarity(v1, v2):
  v1_norm = np.linalg.norm(v1)
  v2_norm = np.linalg.norm(v2)
  if v1_norm * v2_norm == 0:
    return 1.0
  return np.dot(v1, v2) / (v1_norm * v2_norm)


model = "text-embedding-3-small"

for file in os.listdir("new_results/"):
    data = read_json("new_results/"+file)
    print("Working on ", file)

    for item in data:
        inp = [item['answer']] + [answer for answer in item['pred_diverse_ans']]
        inp = [remove_punc_and_lower(ans) for ans in inp]
        orig_emb = np.array(client.embeddings.create(input = [inp[0]], model=model).data[0].embedding)

        inp = [remove_punc_and_lower(ans) for ans in inp]
        openai_resp = client.embeddings.create(input = inp, model=model)
        emds = [np.array(openai_resp.data[i].embedding) for i in range(len(inp))]

        orig_emb = emds[0]

        item['embedding_model'] = model
        item['embedding_similiarity'] = [cosine_similarity(orig_emb, vec) for vec in emds[1:]]

    write_json(data, f"new_results/{file}")


