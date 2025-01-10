FROM python:3.8-slim-buster
WORKDIR /service
COPY  recuirement.txt .
COPY . ./
RUN pip install -r recuirements.txt
ENTRYPOINT ["python3","app.py" ]