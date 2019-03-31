import pymongo
import json

client = pymongo.MongoClient(host='10.100.99.85')
db = client.vca
qanda_collection = db.qanda

with open('../train.json') as f:
    data_json = json.load(f)

assert data_json is not None



# new_qanda = db.new_qanda

qanda_all = qanda_collection.find()

all_q = []
_class = None
for q in qanda_all:
    if _class is None and q['_class'] != None:
        _class = q['_class']
    del q['_id']
    if 'text' in q:
        del q['text']
    all_q.append(q)

assert len(all_q) != 0

for d in data_json['data']:
    d['_class'] = _class

all_q = all_q + data_json['data']

qanda_collection.remove({})
qanda_collection.insert_many(all_q)
