FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY p7_global p7_global
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
RUN make all

EXPOSE 80
CMD uvicorn p7_global.api.app:app --host 0.0.0.0 --port $PORT
