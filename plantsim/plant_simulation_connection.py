# client

import socket
import json


def connect_to_server(ip_address, server_port):
    client_socket = socket.socket(socket.AF_INET,  # ... Netzwerkprotokoll
                                  socket.SOCK_STREAM)  # Protokollart TCP
    server_address = (ip_address, server_port)
    client_socket.connect(server_address)
    return client_socket


class PlantSimulationConnection:

    def __init__(self, ip_address='127.0.0.1', server_port=30000):
        self.client_socket = connect_to_server(ip_address, server_port)

    def receive_messages(self):
        messages = []
        decoded_message = ""
        # receive message
        message = self.client_socket.recv(2048)
        # empfangene Nachricht von Bytes umwandeln
        received_message = (message, "utf8")
        # Nachricht von Tupel in string umwandeln
        decoded_message += received_message[0].decode()
        # decoded_message in einzelne Nachrichten splitten
        if "|" in decoded_message:
            decoded_message = decoded_message.split("|")
            for message_part in decoded_message:
                if message_part != "":
                    messages.append(json.loads(message_part))
            #print(json_message)
        return messages

    def send_message(self, json_message):
        message = json.dumps(json_message)  # to string
        #print('send_message: ', message)
        self.client_socket.send(bytes(message, "utf8"))
