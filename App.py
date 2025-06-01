import streamlit as st
import pickle
import pandas as pd
from datetime import date, timedelta
import statsapi
import joblib

model = joblib.load('best_pitch_model.pkl')

schedule = statsapi.schedule(start_date=date.today(), end_date=date.today(), team="135",sportId=1)
if schedule:
    game_id = schedule[0]['game_id']

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
pitcher_ids = [p['id'] for p in pitchers]


def get_next_padres_pitch_context(game_id, padres_pitcher_ids):
    live_feed = statsapi.get('game_playByPlay', {'gamePk': game_id})
    all_plays = live_feed.get('allPlays', [])
    current_play = all_plays[-1]

    play_events = current_play.get('playEvents', [])

    if play_events:
        prev_pitch_type = play_events[-1].get('details', {}).get('type', {}).get('code', 'firstPitch')
    else:
        prev_pitch_type = 'firstPitch'

    
    features = {
        'pitcher': current_play['matchup'].get('pitcher', {}).get('id', 'None'),
        'batter': current_play['matchup'].get('batter', {}).get('id', 'None'),
        'on_1b': int('postOnFirst' in current_play['matchup']),
        'on_2b': int('postOnSecond' in current_play['matchup']),
        'on_3b': int('postOnThird' in current_play['matchup']),
        'if_fielding_alignment': current_play['matchup'].get('ifFieldingAlignment', 'standard'),
        'of_fielding_alignment': current_play['matchup'].get('ofFieldingAlignment', 'standard'),
        'prev_pitch_type': prev_pitch_type,
        'inning':  current_play['about'].get('inning'),
        'balls':  current_play['count'].get('balls'),
        'strikes': current_play['count'].get('strikes'),
        'outs_when_up': current_play['count'].get('outs'),
        'score_diff': abs(current_play['result'].get('homeScore') - current_play['result'].get('awayScore'))
                }

    return features

data = get_next_padres_pitch_context(game_id, pitcher_ids)
if data is None:
    data = features = {
                    'pitcher': '605397',
                    'batter': '606466',
                    'on_1b': 0,
                    'on_2b': 0,
                    'on_3b': 0,
                    'if_fielding_alignment': 'standard',
                    'of_fielding_alignment': 'standard',
                    'prev_pitch_type': 'FF',
                    'inning': 1,
                    'balls': 0,
                    'strikes': 0,
                    'outs_when_up': 0,
                    'score_diff': 0
                }
input_df = pd.DataFrame([data])

encode_dict = {0: 'CH',
 1: 'CU',
 2: 'FC',
 3: 'FF',
 4: 'FS',
 5: 'KC',
 6: 'SI',
 7: 'SL',
 8: 'ST'}

code_to_name = {'ST': 'Sweeper',
 'CH': 'Changeup',
 'FF': 'Four Seam Fastball',
 'SI': 'Sinker',
 'SL': 'Slider',
 'FC': 'Cutter',
 'CU': 'Curveball',
 'KC': 'Knuckleball',
 'FS': 'Split-finger'}

pitcher_info = statsapi.get('person', {'personId': data['pitcher']})
pitcher_name = pitcher_info['people'][0]['fullName']
batter_info = statsapi.get('person', {'personId': data['batter']})
batter_name = batter_info['people'][0]['fullName']

prediction = model.predict(input_df)
pitch_type = encode_dict.get(prediction[0])
name_of_pitch = code_to_name.get(pitch_type)
print("Predicted pitch type:", name_of_pitch)

st.title("Pitch Type Predictor")

if st.button("Predict Pitch Type"):
    prediction = model.predict(input_df)
    pitch_type = encode_dict.get(prediction[0])
    name_of_pitch = code_to_name.get(pitch_type)
    st.write(f"Predicted pitch type: {name_of_pitch}")
    st.write(f"Inputs: {data}")
    st.write(f"Pitcher Name: {pitcher_name}")
    st.write(f"Batter Name: {batter_name}")


