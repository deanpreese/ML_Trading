from common.common_cli import  PulsarConsumerCli

pulsar_cli = PulsarConsumerCli()
consumer = pulsar_cli.get_consumer('tester')

while True:
    msg = consumer.receive()
    try:
        print("Received message '{}' id='{}'".format(msg.data(), msg.message_id()))
        # Acknowledge successful processing of the message
        consumer.acknowledge(msg)
    except Exception:
        # Message failed to be processed
        consumer.negative_acknowledge(msg)

pulsar_cli.client.close()