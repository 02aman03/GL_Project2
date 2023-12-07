FROM python:3.11
WORKDIR /pythonProject/
COPY . /pythonProject/
RUN pip install -r requirements.txt
RUN python main.py