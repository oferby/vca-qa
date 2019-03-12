# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: question.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='question.proto',
  package='question',
  syntax='proto3',
  serialized_options=_b('\n\027com.huawei.vca.questionB\rQuestionProtoP\001'),
  serialized_pb=_b('\n\x0equestion.proto\x12\x08question\"7\n\x0fQuestionRequest\x12\x10\n\x08question\x18\x01 \x01(\t\x12\x12\n\nparagraphs\x18\x02 \x03(\t\"J\n\x10QuestionResponse\x12\x0e\n\x06\x61rgmax\x18\x01 \x01(\x05\x12\x11\n\tparagraph\x18\x02 \x01(\t\x12\x13\n\x0bprobability\x18\x03 \x03(\x02\x32\x61\n\x0fQuestionService\x12N\n\x13getQuestionResponse\x12\x19.question.QuestionRequest\x1a\x1a.question.QuestionResponse\"\x00\x42*\n\x17\x63om.huawei.vca.questionB\rQuestionProtoP\x01\x62\x06proto3')
)




_QUESTIONREQUEST = _descriptor.Descriptor(
  name='QuestionRequest',
  full_name='question.QuestionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='question', full_name='question.QuestionRequest.question', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='paragraphs', full_name='question.QuestionRequest.paragraphs', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=83,
)


_QUESTIONRESPONSE = _descriptor.Descriptor(
  name='QuestionResponse',
  full_name='question.QuestionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='argmax', full_name='question.QuestionResponse.argmax', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='paragraph', full_name='question.QuestionResponse.paragraph', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='probability', full_name='question.QuestionResponse.probability', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=159,
)

DESCRIPTOR.message_types_by_name['QuestionRequest'] = _QUESTIONREQUEST
DESCRIPTOR.message_types_by_name['QuestionResponse'] = _QUESTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QuestionRequest = _reflection.GeneratedProtocolMessageType('QuestionRequest', (_message.Message,), dict(
  DESCRIPTOR = _QUESTIONREQUEST,
  __module__ = 'question_pb2'
  # @@protoc_insertion_point(class_scope:question.QuestionRequest)
  ))
_sym_db.RegisterMessage(QuestionRequest)

QuestionResponse = _reflection.GeneratedProtocolMessageType('QuestionResponse', (_message.Message,), dict(
  DESCRIPTOR = _QUESTIONRESPONSE,
  __module__ = 'question_pb2'
  # @@protoc_insertion_point(class_scope:question.QuestionResponse)
  ))
_sym_db.RegisterMessage(QuestionResponse)


DESCRIPTOR._options = None

_QUESTIONSERVICE = _descriptor.ServiceDescriptor(
  name='QuestionService',
  full_name='question.QuestionService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=161,
  serialized_end=258,
  methods=[
  _descriptor.MethodDescriptor(
    name='getQuestionResponse',
    full_name='question.QuestionService.getQuestionResponse',
    index=0,
    containing_service=None,
    input_type=_QUESTIONREQUEST,
    output_type=_QUESTIONRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_QUESTIONSERVICE)

DESCRIPTOR.services_by_name['QuestionService'] = _QUESTIONSERVICE

# @@protoc_insertion_point(module_scope)