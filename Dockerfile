FROM public.ecr.aws/lambda/python:3.8

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.4.4-cp38-cp38-linux_x86_64.whl

COPY "models/prediction.tflite" .
COPY lambda_function.py .
COPY utils.py .

CMD ["lambda_function.lambda_handler"]