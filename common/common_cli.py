import json
import requests
import pulsar
from kafka import KafkaProducer
from kafka import KafkaConsumer

class KafkaProducerCli:
    def __init__(self, **kwargs):
        
        self.producer = KafkaProducer(bootstrap_servers=['10.0.0.50:9092'])
        self.topic = "order_topic"

    def send_data(self, topic, data):
        
        data=json.dumps(data,default=str)
        self.producer.send(topic, data.encode('utf-8'))
        self.producer.flush()
        
    def send_order_data(self, data):
        
        data=json.dumps(data,default=str)
        self.producer.send(self.topic, data.encode('utf-8'))
        self.producer.flush()
        

class KafkaConsumerCli:
    def __init__(self, **kwargs):
        self.topic = "order_topic"
        
        self.consumer = KafkaConsumer('model-orders',
                         group_id='foox',
                         bootstrap_servers=['10.0.0.50:9092'])
        

    def get_order_consumer(self):
        return  self.consumer.subscribe(self.topic)        
                
    def get_consumer(self):
        return  self.consumer 


class PulsarConsumerCli:
    def __init__(self, **kwargs):
        
        PULSAR_NEW_ORDER_TOPIC =  "persistent://public/models/new-model-orders"
        self.topic = PULSAR_NEW_ORDER_TOPIC
        #self.client = pulsar.Client('pulsar://10.0.0.82:6650')
        self.client = pulsar.Client('pulsar://10.0.0.50:6650')
       

    def get_consumer(self, subs):
        return  self.client.subscribe(self.topic, subs)


class PulsarProducerCli:
    def __init__(self, **kwargs):
        
        PULSAR_NEW_ORDER_TOPIC =  "persistent://public/models/new-model-orders"
        self.topic = PULSAR_NEW_ORDER_TOPIC
        #self.client = pulsar.Client('pulsar://10.0.0.82:6650')
        self.client = pulsar.Client('pulsar://10.0.0.50:6650')
        self.producer = self.client.create_producer(self.topic)
                  
    def send_order_pulsar(self, new_order):
        
        #self.producer = self.client.create_producer(self.topic)
        data=json.dumps(new_order,default=str)
        self.producer.send(data.encode('utf-8'))


class CommonCli:
               
    def __init__(self, client_url):               
        #self.base_url = "http://10.0.0.147:8786/"
        self.base_url = client_url

    def initialize_trader(self,display_name, group_num ):
        
        t_id = 0
        
        url = self.base_url + "user/verify-model-trader"  

        headers = {
            "Content-Type": "application/json"
        }

        add_trader_model = {  
            "groupId": group_num,
            "userId": 0,
            "displayName": display_name,
            "userPwd": "abc",
            "firstName": "Model",
            "lastName": "Trader",
            "email": "bac.abc"
        }
                        
        response = requests.post(url, data=json.dumps(add_trader_model), headers=headers)

        if response.status_code == 200:
            result = response.json()
            t_id = result
            
        else:
            # Error handling
                print(f"Trader initialize request failed with status code {response.status_code}")

        return t_id
                        
  
                    

    def send_order(self, new_order):
        
        url = self.base_url + "order/ml/process-order"  

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, data=json.dumps(new_order,default=str), headers=headers)
        if response.status_code == 200:
            pass
        else:
            print(f"New Order send failed with status code {response.text}")                


    def send_order_z(self, new_order):
        
        url = self.base_url + "order/ml/process-order-z"  

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, data=json.dumps(new_order,default=str), headers=headers)
        
        if response.status_code == 200:
            pass
        else:
            print(f"New Order send failed with status code {response.text}")                          