#import json
#import requests
from enum import Enum
import datetime as dt

class position_status(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class order_action(Enum):
    Buy = 2
    Sell = -2
    Ignore = 0

class order_type(Enum):
    Open = 100
    Close = 99
    NOTYPE = -1



class CommonStrategy :
    
   def __init__(self):
        
      self.bars_exit = 5
      self.entry_px = 0
      self.profit = 5
      self.stop = 3
      self.metrics = []
   
      self.last_prediction = 0
      self.position = position_status.FLAT
      self.bars_since = 0
      
      self.perf = 0
      
      self.predict_data = ""
   
   def set_predict_data(self, data):
      self.predict_data = data      

   def check_for_orders(self, prediction, px):
      
      if prediction == 0:
          new_order_action = order_action.Ignore
         
      if prediction > 0:
         new_order_action = order_action.Buy
      if prediction < 0:
         new_order_action = order_action.Sell

      orders = []

      if self.position == position_status.FLAT:
         orders.append(self.build_order( new_order_action, px, order_type.Open)) 

         if new_order_action == order_action.Buy:
               self.position = position_status.LONG 

         if new_order_action == order_action.Sell:
               self.position = position_status.SHORT

         self.last_prediction = prediction
         self.bars_since = 1
         
         return orders
         
      if self.position == position_status.LONG:
         
         # already long dont add
         if  new_order_action == order_action.Buy:
               self.last_prediction = prediction
               self.bars_since += 1
         
               if self.check_long_pl(px) or self.check_bars_since(px) :
                  print(f"Close Long PnL or bars {self.bars_exit}  {px}") 
                  orders.append(self.close_this_order(px))
         
         # long but sell order
         if new_order_action == order_action.Sell:
               orders.append(self.build_order(new_order_action, px, order_type.Close))  
               self.last_prediction = prediction
               self.position = position_status.FLAT
               self.bars_since = 0
               
         return orders
      
      if self.position == position_status.SHORT:

         # already short dont add
         if  new_order_action == order_action.Sell:
               self.last_prediction = prediction
               self.bars_since += 1
               
               if self.check_short_pl(px) or self.check_bars_since(px) :
                  print(f"Close Long PnL or bars {self.bars_exit}  {px}") 
                  orders.append(self.close_this_order(px))
         
         # short but buy order
         if new_order_action == order_action.Buy:
               orders.append(self.build_order(new_order_action, px, order_type.Close))  
               self.last_prediction = prediction
               self.position = position_status.FLAT
               self.bars_since = 0


         return orders  
      
      return orders                      
   
   
   def check_bars_since(self, px):
        
        if self.bars_since > self.bars_exit:
            print(f"Close bars greater than {self.bars_exit}  {px}") 
            return True
        return False
        
   def check_short_pl(self, px):
      
      if self.entry_px > px and abs(self.entry_px - px) > self.profit: 
         return True
      
      if self.entry_px < px and abs(self.entry_px - px) > self.stop:
         return True
      
      return False    
   
   def check_long_pl(self, px):
      
      if self.entry_px < px and abs(self.entry_px - px) > self.profit: 
         return True
      
      if self.entry_px > px and abs(self.entry_px - px) > self.stop:
         return True
   
      return False
   
   


   def close_this_order(self, px):

      if self.position == position_status.LONG:
         self.position = position_status.FLAT
         self.bars_since = 0
         return self.build_order(order_action.Sell,px,order_type.Close)
         
      if self.position == position_status.SHORT:
         self.position = position_status.FLAT
         self.bars_since = 0  
         return self.build_order(order_action.Buy,px,order_type.Close)
           
    
   # --------------------------------------
   #  Called by OM to close all open orders    
   def close_orders(self, px):
      orders_to_close = []
      if self.position == position_status.LONG:
         orders_to_close.append( self.build_order(order_action.Sell,px, order_type.Close))
         self.position = position_status.FLAT
         
      if self.position == position_status.SHORT:
         orders_to_close.append( self.build_order(order_action.Buy,px, order_type.Close))
         self.position = position_status.FLAT  
      return orders_to_close
   
   
   def build_order(self, new_order_action, px, new_order_type):
      
      dt_string = dt.datetime.utcnow()
      dte_iso = dt_string.isoformat()
      self.entry_px = px
      
      new_order = {
         "newOrderID": 0,
         "platformOrderID": 0,
         "userName": self.run_name,
         "userID": int(self.trader_id),
         "groupID": int(self.trader_group),
         "authToken": 0,
         "instrument": "ES",
         "orderPX": px,
         "orderType": int(new_order_type.value),
         "orderAction": int(new_order_action.value),
         "quantity": 1,
         "orderTime": dte_iso,
         "modelFeatureData" :  self.predict_data.to_json()
      }    

      
      return new_order