#gRPC

predict answer for user question using BERT

create gRCP objects from PROTO files
  
pip install grpcio
pip install grpcio-tools

python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./question.proto

