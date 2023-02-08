import torch
import numpy as np
import time
import datasets
import random
import argparse
import json

from kmeans import product_quantization, data_to_pq
import pandas as pd
from sentence_transformers import SentenceTransformer


def read_query(file_name):
    df = pd.read_csv(file_name, sep='\t', names=["query_id", 'query'])
    dict = {}
    for i, row in df.iterrows():
        dict[row['query_id']] = row['query']
    return dict


def read_qrel(file_name):
    dict = {}
    df = pd.read_csv(file_name, sep='\t', names=["query_id", '2', 'doc_id', '4'])
    for i, row in df.iterrows():
        query_id = row['query_id']
        doc_id = row['doc_id']
        doc_id = int(doc_id)
        if doc_id not in dict:
            dict[doc_id] = [query_id]
        else:
            dict[doc_id].append(query_id)
    return dict


def get_pq(sentence_embeddings):
    training_data = torch.from_numpy(sentence_embeddings)

    if torch.cuda.is_available():
        training_data = training_data.cuda()

    codebook = product_quantization(
        training_data,
        subv_size,
        k=num_centers, iter=2,
        batch_size=128)

    centers = torch.stack(codebook).cpu().numpy()
    print(centers.shape)

    if torch.cuda.is_available():
        codebook = [i.cuda() for i in codebook]

    pq_data = data_to_pq(training_data, codebook)

    pq = pq_data.tolist()

    return pq


torch.set_num_threads(16)
subv_size = 24
num_centers = 256


data = datasets.load_dataset('Tevatron/msmarco-passage-corpus', data_files='./data/corpus.jsonl', cache_dir='cache')[
    'train']
parser = argparse.ArgumentParser()
parser.add_argument("--train_num", type=int)
parser.add_argument("--eval_num", type=int, default=6980)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
NUM_TRAIN = args.train_num
NUM_EVAL = args.eval_num

model = SentenceTransformer('uer/sbert-base-chinese-nli')
print("Creating MS MARCO dataset...")
dev_query = read_query('msmarco_data/dev.query.tsv')
dev_qrel = read_qrel('msmarco_data/qrels.dev.small.tsv')
train_query = read_query('msmarco_data/train.query.tsv')
train_qrel = read_qrel('msmarco_data/qrels.train.tsv')

train_ids = list(train_qrel.keys())
random.shuffle(train_ids)
train_ids = train_ids[:NUM_TRAIN]
dev_ids = list(set(dev_qrel.keys()).difference(set(train_qrel.keys())))  # make sure no data leakage
random.shuffle(dev_ids)
dev_ids = dev_ids[:NUM_EVAL]


# sentences = train_query_df['2'].tolist()
# sentence_embeddings = model.encode(sentences[:256])
DSI_train_data = []
DSI_dev_data = []
corpus_data = []


current_ind = 0
current_pq = 0
passages = []
for docid in train_ids:
    passage = data[docid]['text']
    passages.append(passage)

df = pd.DataFrame(data=passages, columns=['passage'])
sentence_embeddings = model.encode(df['passage'].tolist())
pq_list = get_pq(sentence_embeddings)
for docid in train_ids:
    passage = data[docid]['text']

    current_pq = pq_list[current_ind]
    question = train_query[train_qrel[docid][0]]

    DSI_train_data.append({'text_id': current_pq, 'text': 'Passage: ' + passage})
    DSI_train_data.append({'text_id': current_pq, 'text': 'Question: ' + question})
    corpus_data.append(f"{current_pq}\t{passage}")
    current_ind += 1

# for item in corpus:
#     if current_ind >= NUM_TRAIN:
#         break
#     passage = item['text']
#     DSI_train_data.append({'text_id': current_pq,
#                            "text": f"Passage: {passage}"})
#     corpus_data.append(f"{current_pq}\t{passage}")
#     current_ind += 1

for docid in dev_ids:
    passage = data[docid]['text']
    question = dev_query[dev_qrel[docid][0]]

    if len(DSI_dev_data) < NUM_EVAL:
        DSI_train_data.append({'text_id': current_pq,
                               "text": f"Passage: {passage}"})
        DSI_dev_data.append({'text_id': current_pq,
                             "text": f"Question: {question}"})
        corpus_data.append(f"{current_pq}\t{passage}")
        current_ind += 1

with open(f'{args.save_dir}/msmarco_DSI_train_data.json', 'w') as tf, \
        open(f'{args.save_dir}/msmarco_DSI_dev_data.json', 'w') as df:
    [tf.write(json.dumps(item) + '\n') for item in DSI_train_data]
    [df.write(json.dumps(item) + '\n') for item in DSI_dev_data]

with open(f'{args.save_dir}/msmarco_corpus.tsv', 'w') as f:
    [f.write(item + '\n') for item in corpus_data]
# db = np.random.randn(db_size, dim)
