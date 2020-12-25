FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && apt-get install -y \
  gdb  \
  gdbserver \
  python3.8-dbg 

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY ./adeptRL ./
COPY ./sc2 ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install adeptRL[mpi,sc2,profiler]

COPY *.py ./
