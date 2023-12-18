FROM public.ecr.aws/lambda/python:3.8

COPY tflite_runtime-2.4.4-cp38-cp38-linux_x86_64.whl .

RUN pip install keras-image-helper
RUN pip install tflite_runtime-2.4.4-cp38-cp38-linux_x86_64.whl

COPY prediction.tflite .
COPY lambda_function.py .
COPY utils.py .

CMD ["lambda_function.lambda_handler"]