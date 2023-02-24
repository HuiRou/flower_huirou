#!/bin/bash

client_num=8
comm_round=2


echo "Starting server"
python server.py $comm_round&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 `expr $client_num - 1`); do
    echo "Starting client $i"
    python client.py $client_num $i&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
