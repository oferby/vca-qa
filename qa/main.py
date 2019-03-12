from concurrent import futures
import time

import grpc
import qa.proto.question_pb2 as pb2
import qa.proto.question_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ParagraphFinder(pb2_grpc.QuestionServiceServicer):
    def getQuestionResponse(self, request, context):
        print('got new request: {}'.format(request))
        return pb2.QuestionResponse(paragraph='echo: {}'.format(request.question))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_QuestionServiceServicer_to_server(ParagraphFinder(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


serve()
