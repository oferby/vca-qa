from elasticsearch import Elasticsearch
import json

es = Elasticsearch([
    {'host': '10.100.99.85'}
    # {'host': '18.222.131.22'}
])

print(es.info())

import pymongo
import json

client = pymongo.MongoClient(host='10.100.99.85')
db = client.vca
qanda_collection = db.qanda


# with open('../train.json') as f:
#     data_json = json.load(f)

# assert data_json is not None

es.indices.delete(index='qa', ignore=[400, 404])

qanda_all = qanda_collection.find()

i = 1
for p in qanda_all:
    del p['_class']
    del p['_id']
    if 'questions_not' in p:
        del p['questions_not']
    es.index(index='qa', doc_type='question', id=i, body=p)
    i += 1

assert qanda_all is not None
