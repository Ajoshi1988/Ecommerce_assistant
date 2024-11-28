# from channels.consumer import SyncConsumer, AsyncConsumer
# from channels.exceptions import StopConsumer

# class MySyncConsumer(SyncConsumer):
#     def websocket_connect(self, event):
#         print('Websocket', event)
        
#         self.send({
#             'type': 'websocket.accept'
#         })
        
#     def websocket_receive(self, event):
#         print('Message.. recieved', event)
    

#     def websocket_disconnect(self, event):
#         print("Message Disconnected", event)