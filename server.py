import sys
import flwr as fl

mode = str(sys.argv[1])
num_rounds = int(sys.argv[2])

# Start Flower server
fl.server.start_server(
    server_address="192.168.50.179:8080",
    config=fl.server.ServerConfig(mode=mode, num_rounds=num_rounds),
)
