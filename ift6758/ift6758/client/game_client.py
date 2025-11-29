# Importing relevant libraries
import requests 
import pandas as pd 
import logging 
from typing import Dict, List, Optional

# Defining Logger 
logger = logging.getLogger(__name__)

# Defining Base_URL derived from Data Acquisition in Milestone 1 and 2 
BASE_URL = "https://api-web.nhle.com/v1/gamecenter"


# Defining game client class
class GameClient:
    """
    A lightweight client for processing events for live NHL games via game_id

    This class will be able to:
    -Fetch JSON for a given game_id 
    -Extracting the relevant plays 
    -Monitoring plays which are already processed 
    -Return only new unseen events in the form of a tidy dataframe
    """

    # Defining constructor method 
    def __init__(self, game_id: str):
        """
        game_id examples:
            Regular season game_id: 202302001
            Playoff gamed_id: 2023030112
        """
        # Defining nhl game_id
        self.game_id = game_id
        # Defining nhl url 
        self.url_api = f"{BASE_URL}/{game_id}/play-by-play"
        # Defining already seen_event_id
        self.already_seen_event_ids: set = set()
        logger.info(f"[GameClient] established for game_id={game_id}")
    
    # Fetching game data through a json file 
    def obtain_game(self) -> Optional[Dict]:
        """Obtaining raw json NHL data, it will return none if nothing is found"""
        try:
            # Request given url with specific game id and return json file 
            r = requests.get(self.url_api)
            r.raise_for_status()
            return r.json()
            # Raise an Exception if there is an error obtaining JSON game 
        except Exception as e:
            logger.error(f"[GameClient] has experienced an error obtaining JSON game.")
            return None
    
    def extract_new_events(self, data: Dict) -> pd.DataFrame:
        """This function extracts only unseen plays and will add new event id to these plays"""
        # If data is None or play not in data return pd.Dataframe
        if data is None or "plays" not in data:
            return pd.DataFrame()
        
        # Defining valid plays
        valid_plays = data["plays"]
        # If not valid plays, return DataFrame 
        if not isinstance(valid_plays, list):
            return pd.DataFrame() 
        # Defining data frame by json normalizing valid plays 
        df = pd.json_normalize(valid_plays)
        
        # Return df if eventId is not in df.columns()
        if "eventId" not in df.columns:
            return pd.DataFrame()
        
        # Keeping only unseen events 
        # Defining eventId as integer 
        df["eventId"] = df["eventId"].astype(int)
        updated_df = df[~df["eventId"].isin(self.already_seen_event_ids)]

        # Update Tracker 
        new_ids = set(updated_df["eventId"].tolist())
        self.already_seen_event_ids.update(new_ids)
        
        # Putting information into the log
        logger.info(f"[GameClient] {len(updated_df)} has a new event extracted.")

        return updated_df
    
    def extract(self) -> pd.DataFrame:
        """
        Fetching json and returning unseen hockey events
        """
        data = self.obtain_game()
        unseen_events = self.extract_new_events(data)
        return unseen_events 


        







        