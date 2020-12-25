FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && apt-get install -y \
  gdb  \
  gdbserver \
  python3.8-dbg 

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sc2 /usr/src/app/sc2
COPY envs /usr/src/app/envs
COPY agents /usr/src/app/agents
COPY *.py ./
COPY .env ./
