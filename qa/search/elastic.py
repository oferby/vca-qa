from elasticsearch import Elasticsearch


class ElasticSearcher:
    def __init__(self):
        self.es = Elasticsearch([
            {'host': '10.100.99.85'}
        ])

    def search_qa(self, text):
        body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": text
                            }
                        },
                        {
                            "match": {
                                "question": text
                            }
                        }
                    ]
                }
            }
        }

        docs = self.es.search(index='qa', doc_type='question', body=body)
