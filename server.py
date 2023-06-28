import sys
import flwr as fl

num_rounds = int(sys.argv[1])

# Start Flower server
fl.server.start_server(
    server_address="192.168.50.179:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
)
