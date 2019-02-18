import os
from dotenv import load_dotenv, find_dotenv
from pushbullet import Pushbullet
load_dotenv(find_dotenv())
class PBNotifier():
    """Class to handle PushBullet notifications.
    
    Note: 
        The Pushbullet API key must be specified in a .env file.

    Read-only properties:
        api_key (str): Pushbullet API key.
    """

    
    def __init__(self):
        
        self._api_key = os.getenv("PUSHBULLET_API_KEY")
        pass

    def send_message(self, title:str, body:str):
        """Send a message using Pushbullets API.
        
        Args:
            title (str): Subject of message.
            body (str): Body of message.
        """

        assert isinstance(title, str) and isinstance(body, str), 'Title and body must be strings'


        if self._api_key is not None:
            pb = Pushbullet(self._api_key)
            _= pb.push_note(title, body)
        else:
            print('Pushbullet API is not defined')

    def fetch_apikey(self):
        """Get the users API Key
        
        Returns:
            str: API Key
        """

        return self._api_key