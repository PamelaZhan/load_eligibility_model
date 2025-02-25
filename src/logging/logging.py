import logging
# decorator that logs errors when exceptions occur within any function

# Create a logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename="realestate.log", level=logging.INFO, format="[%(asctime)s]  %(levelname)s -- %(message)s")

def logging_decorator(func):
    
    def wrapper(*args, **kwargs):
        try:
            # log function called infomation
            logger.info(f"Function '{func.__name__}' started with arguments {args} and {kwargs}")
            return func(*args, **kwargs)
        # log errors
        except Exception as e:
            logger.error(f"Unexpected error occurred in function '{func.__name__}' with arguments {args} and {kwargs}. Error: {e}")
            return None
        finally:
            # log the end of functions execution
            logger.info(f"Function '{func.__name__}' ended")
    
    return wrapper