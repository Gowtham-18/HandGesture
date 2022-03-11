FROM docker-tensorflow-opencv

WORKDIR /opt/app
COPY . .

RUN apk add --no-cache python3 py3-pip python3-dev g++ gcc
RUN pip install -U pip
RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
