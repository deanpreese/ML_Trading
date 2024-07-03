from common.common_cli import  KafkaConsumerCli


broker_cli = KafkaConsumerCli()
consumer = broker_cli.get_consumer()

while True:
     
    act_val = 0
    pre_val = 0
         
    sum_val = 0
    sum_act = 0
    

    ref2 = 0
     
    consumer.subscribe( "order_topic")    
    
    for message in consumer:
            
            pluses = 0
            minuses = 0
    
            
            decoded_string = message.value.decode('utf-8')
            cleaned_string = decoded_string.strip('"')
            number_strings = cleaned_string.split(',')
            #print(number_strings[0] +  "    "  + number_strings[1])
            
            
            pre_val = float(number_strings[0])
            act_val = float(number_strings[1])            
            
            ref2 += pre_val + act_val
            
            if pre_val > 0:
                pluses += 1
            else:
                minuses += 1
            
            sum_val += pre_val
            sum_act += act_val
            
            print(f"Totals {sum_act}    { sum_val }  {ref2}   {pre_val}   {act_val}  ++  {pluses}       -- {minuses}")
            