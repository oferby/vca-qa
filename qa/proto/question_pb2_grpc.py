# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import qa.proto.question_pb2 as question__pb2


class QuestionServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.getQuestionResponse = channel.unary_unary(
        '/question.QuestionService/getQuestionResponse',
        request_serializer=question__pb2.QuestionRequest.SerializeToString,
        response_deserializer=question__pb2.QuestionResponse.FromString,
        )


class QuestionServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def getQuestionResponse(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_QuestionServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'getQuestionResponse': grpc.unary_unary_rpc_method_handler(
          servicer.getQuestionResponse,
          request_deserializer=question__pb2.QuestionRequest.FromString,
          response_serializer=question__pb2.QuestionResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'question.QuestionService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
