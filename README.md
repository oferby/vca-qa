# gRPC

predict answer for user question using BERT

create gRCP objects from PROTO files
  
pip install grpcio
pip install grpcio-tools

python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./question.proto


# To start the server on AWS g3s.xlarge instance

run:

    conda activate tensorflow_p36  
    python main.py


    {
      "paragraph": "",
      "questions": [
        ""  
      ]
    }
