from datetime import datetime as dt
from datetime import timedelta

from common.common_cli import CommonCli, KafkaProducerCli 

class OrderManager():
        
    
    def __init__(self, api_url):
        self.px = 0
        self.commission = 0
        self.tick_dte = dt(2020, 1, 1)
        
        self.com_cli = CommonCli(api_url)
        #self.pulsar_cli = PulsarProducerCli()
        #self.kafka_cli = KafkaProducerCli()
        

    def is_sim(self, is_sim_yn):
        self.is_sim_yn = is_sim_yn
        
        if is_sim_yn is True:
            self.px = 5000
        

    def process_tick(self, tick):
        self.px += tick    
        self.tick_dte = self.tick_dte + timedelta(minutes=5)
        

    def process_tick_rt(self, tick, time_ticks):
        self.px = tick    
        self.tick_dte = time_ticks
    
    def process_model_ninja_data(self, model, y_pred, prediction):
        
        out_orders = []
        orders = model.check_for_orders(prediction, self.px)        
        
        if len(orders) > 0:        
             for ord in orders:
                ord['orderTime']  = self.tick_dte
                out_orders.append(ord)
 
        return out_orders
        
    def process_model(self, model, y_pred, prediction):

        orders = model.check_for_orders(prediction, self.px)        
        
        if len(orders) > 0:        
             for o in orders:
                self.send_order(o)
 

    def close_all(self, model):
        
        o_t_c = model.close_orders(self.px)
        
        if len(o_t_c) > 0:
             for o in o_t_c:
                self.send_order(o)


    def send_order(self, new_order):

        new_order['orderTime']  = self.tick_dte

        if self.is_sim_yn is True:
            new_order['orderTime']  = self.tick_dte.isoformat()
                
        if self.px > 0:
        
            print(f"X  Order   {new_order['userID']}   {new_order['userName']}   {new_order['orderAction']}  {new_order['orderPX']} {new_order['orderTime']}" ) 
    
            self.com_cli.send_order(new_order)
            #self.com_cli.send_order_z(new_order)
            #self.pulsar_cli.send_order_pulsar(new_order) 
            #self.kafka_cli.send_order_data(new_order['modelFeatureData'])    
            
               
            
