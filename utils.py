from configparser import ConfigParser

def read_config():
    config = ConfigParser()
    config.read('config.ini')

    return config
