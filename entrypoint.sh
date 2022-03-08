#!/bin/sh

python app.py &
ssh -R 80:localhost:8080 nokey@localhost.run