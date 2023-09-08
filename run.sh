#!/bin/bash

client_num=8
comm_round=100
mode=train
#mode=test
#mode=avg
#mode=random


echo "Starting server"
python server.py $mode $comm_round&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 `expr $client_num - 1`); do
    echo "Starting client $i"
    python client.py $mode $client_num $i&
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
