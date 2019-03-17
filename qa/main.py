from concurrent import futures
import time
import queue
import threading

import grpc
import qa.proto.question_pb2 as pb2
import qa.proto.question_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ParagraphEcho(pb2_grpc.QuestionServiceServicer):
    def getQuestionResponse(self, request, context):
        print('got new request:\n{}'.format(request))
        return pb2.QuestionResponse(paragraph='echo: {}'.format(request.paragraphs[0]))


class ThreadedParagraphFinder(pb2_grpc.QuestionServiceServicer):
    def __init__(self, main_q):
        self.main_q = main_q
        super(ThreadedParagraphFinder, self).__init__()

    def getQuestionResponse(self, request, context):
        callback_q = queue.Queue()
        req = {
            "request": request,
            "callback_q": callback_q
        }
        self.main_q.put(req)
        response = callback_q.get(True)
        return pb2.QuestionResponse(paragraph=response['response'])


class Predictor(threading.Thread):
    def __init__(self, main_q):
        self.main_q = main_q
        super(Predictor, self).__init__()

    def run(self):
        while True:
            task = self.main_q.get(True)
            callback_q = task['callback_q']
            response = {
                "response": "thread echo: " + task['request'].paragraphs[0]
            }
            callback_q.put(response)


def serve():
    main_q = queue.Queue()
    predictor = Predictor(main_q)
    predictor.start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_QuestionServiceServicer_to_server(ThreadedParagraphFinder(main_q), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


serve()
