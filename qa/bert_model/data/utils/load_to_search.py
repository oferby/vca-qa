from elasticsearch import Elasticsearch
import json

es = Elasticsearch([
    {'host': '10.100.99.85'}
])

print(es.info())

with open('../train.json') as f:
    data_json = json.load(f)

assert data_json is not None

es.indices.delete(index='qa', ignore=[400, 404])

i = 1
for p in data_json['data']:
    es.index(index='qa', doc_type='question', id=i, body=p)
    i += 1

assert data_json is not None
