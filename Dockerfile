from python:alpine

WORKDIR /opt/app
COPY . .

RUN apk run --no-cache python py3-pip
RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]