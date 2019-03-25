from elasticsearch import Elasticsearch

es = Elasticsearch([
    {'host': '10.100.99.85'}
])

print(es.info())

doc = es.get(index='qa', doc_type='question', id=1)
print('search by id:',doc)

print('********')

body = {
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "paragraph": "ping my ecs"
          }
        },
        {
          "match": {
            "questions": "How can I ping my ecs?"
          }
        }
      ]
    }
  }
}

docs = es.search(index='qa', doc_type='question', body=body)
print(docs)
