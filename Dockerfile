FROM registry.manofsteel0007.duckdns.org/hand-gesture

WORKDIR /opt/app
COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
