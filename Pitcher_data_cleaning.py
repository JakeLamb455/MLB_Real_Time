import pandas as pd
import numpy as np
df = pd.read_csv("DataSets/padres_pitch_data.csv")
df = df[['pitch_type', 'game_date', 'pitcher', 'batter', 'player_name', 'game_pk', 'pitch_number', 'at_bat_number','inning', 'balls', 'strikes', 'zone', 'outs_when_up', 'inning_topbot', 'fld_score', 'bat_score', 'on_1b', 'on_2b', 'on_3b', 'if_fielding_alignment', 'of_fielding_alignment', 'plate_x', 'plate_z']]
df[['pitcher', 'batter', 'game_pk']] = df[['pitcher', 'batter', 'game_pk']].astype('Int64')
df_filt = df.dropna(subset=['pitch_type', 'zone'])
df_filt = df_filt.sort_values(by=['game_pk','pitcher', 'at_bat_number', 'pitch_number'],
                             ascending=True)
df_filt['pitch_count'] = df_filt.groupby(['game_pk','pitcher']).cumcount()+1
df_filt = df_filt.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'], ascending = True)

df_filt['prev_pitch_type'] = (
    df_filt.groupby(['game_pk', 'at_bat_number', 'pitcher'])['pitch_type']
    .shift(1)
)
df_filt['prev_pitch_type'] = df_filt['prev_pitch_type'].fillna('firstPitch')
df_filt[['on_1b', 'on_2b', 'on_3b']] = df_filt[['on_1b', 'on_2b', 'on_3b']].notna().astype(int)
df_filt['score_diff'] = df_filt['fld_score'] - df_filt['bat_score']

if __name__ == "__main__":
    print("Data cleaning complete.")
    df_filt.to_csv('DataSets/pitcher_data_cleaned.csv')