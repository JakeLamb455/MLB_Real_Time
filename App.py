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

    for i in range(len(all_plays) - 1):
        play = all_plays[i]
        next_play = all_plays[i + 1]

        if 'pitchIndex' in play:
            pitcher_id = play.get('matchup', {}).get('pitcher', {}).get('id')
            if pitcher_id in padres_pitcher_ids:
                pitch_event = play.get('playEvents', [{}])[-1]
                prev_pitch_type = pitch_event.get('details', {}).get('type', {}).get('code', 'firstPitch')

                state = next_play
                matchup = state.get('matchup', {})
                about = state.get('about')
                count = state.get('count', {})

                if about is None:
                    print(f"Warning: 'about' missing in play index {i+1}")
                    continue  # skip this iteration

                is_top_inning = about.get('isTopInning', True)

                features = {
                    'pitcher': matchup.get('pitcher', {}).get('id', 'None'),
                    'batter': matchup.get('batter', {}).get('id', 'None'),
                    'on_1b': int('first' in state.get('runnersOn', [])),
                    'on_2b': int('second' in state.get('runnersOn', [])),
                    'on_3b': int('third' in state.get('runnersOn', [])),
                    'if_fielding_alignment': state.get('defensiveAlignment', 'standard'),
                    'of_fielding_alignment': state.get('outfieldAlignment', 'standard'),
                    'prev_pitch_type': prev_pitch_type,
                    'inning': about.get('inning', 1),
                    'balls': count.get('balls', 0),
                    'strikes': count.get('strikes', 0),
                    'outs_when_up': count.get('outs', 0),
                    'score_diff': 0
                }

                return features

    return None

data = get_next_padres_pitch_context(game_id, pitcher_ids)
if data is None:
    data = features = {
                    'pitcher': '605397',
                    'batter': '606466',
                    'on_1b': 1,
                    'on_2b': 0,
                    'on_3b': 0,
                    'if_fielding_alignment': 'standard',
                    'of_fielding_alignment': 'standard',
                    'prev_pitch_type': 'FF',
                    'inning': 3,
                    'balls': 2,
                    'strikes': 2,
                    'outs_when_up': 1,
                    'score_diff': 3
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