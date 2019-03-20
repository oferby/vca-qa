from concurrent import futures
import time
import queue

import grpc
import qa.proto.question_pb2 as pb2
import qa.proto.question_pb2_grpc as pb2_grpc
import qa.bert_model.predit as predict

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# class ParagraphEcho(pb2_grpc.QuestionServiceServicer):
#     def getQuestionResponse(self, request, context):
#         print('got new request:\n{}'.format(request))
#         return pb2.QuestionResponse(paragraph='echo: {}'.format(request.paragraphs[0]))


class ThreadedParagraphFinder(pb2_grpc.QuestionServiceServicer):
    def __init__(self, main_q, thread_q):
        self.main_q = main_q
        self.thread_q = thread_q
        super(ThreadedParagraphFinder, self).__init__()

    def getQuestionResponse(self, request, context):
        print('got new QA request')
        callback_q = self.thread_q
        self.main_q.put(request)
        result = callback_q.get(True)
        print('result: {}'.format(result['probabilities']))
        response = 'response'
        probabilities = []
        p_max = 0
        i_max = 0
        for i, p in enumerate(result['probabilities']):
            if p[1] > p_max:
                p_max = p[1]
                i_max = i
            probabilities.append(float(p[1]))
        print('prob: {}'.format(probabilities))
        return pb2.QuestionResponse(paragraph=response, probability=probabilities, argmax=i_max)


def serve():
    thread_q = queue.Queue(maxsize=1)
    main_q = queue.Queue(maxsize=1)
    predictor = predict.Predictor(main_q, thread_q)
    predictor.start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_QuestionServiceServicer_to_server(ThreadedParagraphFinder(main_q, thread_q), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('gRPC server started')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def train():
    predictor = predict.Predictor()
    predictor.train()



serve()
# train()
