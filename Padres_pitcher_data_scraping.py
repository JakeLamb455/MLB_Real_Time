import pandas as pd
import numpy as np
from datetime import date, datetime
import statsapi


def get_padres_pitchers():
    team_id = 135  # San Diego Padres
    roster = statsapi.get('team_roster', {'teamId': team_id})
    
    pitchers = []
    for player in roster.get('roster', []):
        if player['position']['type'] == 'Pitcher':
            pitchers.append({
                'id': player['person']['id'],
                'name': player['person']['fullName']
            })
    return pitchers

pitchers = get_padres_pitchers()

import pandas as pd
from pybaseball import statcast_pitcher
from datetime import datetime

def fetch_pitcher_data(player_id, start_date="2022-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Fetching data for player {player_id} from {start_date} to {end_date} ...")
    try:
        data = statcast_pitcher(start_date, end_date, player_id)
        return data
    except Exception as e:
        print(f"Error fetching data for player {player_id}: {e}")
        return pd.DataFrame()

def fetch_all_pitchers_data(pitchers, start_date="2022-01-01", end_date=None):
    all_data = []
    for pitcher in pitchers:
        df = fetch_pitcher_data(pitcher['id'], start_date, end_date)
        if not df.empty:
            df['pitcher_name'] = pitcher['name']  # add name for reference
            all_data.append(df)
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total pitches fetched: {len(combined)}")
        return combined
    else:
        print("No data fetched for any pitcher.")
        return pd.DataFrame()

if __name__ == "__main__":
    data = fetch_all_pitchers_data(pitchers)
    data.to_csv("padres_pitch_data.csv", index=False)
    print("Saved pitch data to padres_pitch_data.csv")
