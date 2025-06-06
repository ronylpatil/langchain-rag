# ============== Note: Not in use ==============

import os
import pika
import json
from api.logger import infologger
from dotenv import load_dotenv

load_dotenv(override=True)
AMQP_URL = os.getenv("AMQP_URL")

def connect() -> pika.BlockingConnection:
    """
    Connect to RabbitMQ server using the AMQP URL from environment variables.
    """
    try:
        connection = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
    except Exception as e:
        infologger.info("Failed to connect to RabbitMQ service.")
        raise
    else:
        return connection


def publish(queue: str, message: str) -> None:
    connection = connect()
    channel = connection.channel()  # Create a channel to talk with RabbitMQ
    channel.queue_declare(queue=queue)  # Make sure the queue exists
    channel.basic_publish(
        exchange="",  # Default exchange
        routing_key=queue,  # Queue name = routing key
        body=json.dumps(message),  # Convert dict to JSON string
    )
    infologger.info(f"Data sent to {message['stage']} stage.")
    connection.close()  # Close the connection after sending the message

"""
You mean rabbitmq can scale up/scale down each component independently 

| Feature             | Behavior                                         |
| ------------------- | ------------------------------------------------ |
| Independent scaling | ✅ Each component can scale up/down independently |
| Sequential flow     | ✅ Messages still follow defined service order    |
| Fault isolation     | ✅ If one step breaks, others still run           |
| Load balancing      | ✅ RabbitMQ spreads work across all consumers     |

"""

