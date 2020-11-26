##Import Models and Libs
import numpy as np
import pandas as pd
from datetime import datetime

def convert_date(my_date):
    res = datetime.now().date()
    try:
        res = datetime.strptime(my_date, '%d %b %Y').date()
    except:
        pass
    return res

def import_process_df(path):

    ## Our Main DataFrame

    data = pd.read_pickle(path)

    ## PreProcess or DataFrame

    data = data[data.Odd1.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]               # Convert Our Data To Numerics
    data = data[data.Odd2.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    data = data[data.Winner.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]


    data.reset_index()                                                                                  # Add additional columns for easier prediction
    data['favorite_won'] = 0
    data['favorite_odd'] = 0
    data.loc[(data['Odd1']<=data['Odd2']) & (data['Winner']==1),'favorite_won'] = 1
    data.loc[(data['Odd1']>data['Odd2']) & (data['Winner']==2),'favorite_won'] = 1
    data['favorite_odd']= data[["Odd1","Odd2"]].min(axis=1)
    data['nonfavorite_odd']= data[["Odd1","Odd2"]].max(axis=1)
    data['FullDate'] = data['FullDate'].apply(convert_date)
    ## Normalizing

    df_final = data.copy()
    df_final['favorite_odd_normalized'] = (df_final['favorite_odd']- df_final['favorite_odd'].min())/(df_final['favorite_odd'].max()- df_final['favorite_odd'].min())
    return df_final