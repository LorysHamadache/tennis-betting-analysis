## Import Models and Libs

import models
import data_processing
from datetime import date
import pandas as pd
## Processed Data Frame Import

df_atp_simple = data_processing.import_process_df("./pickles/atp_simples.pkl")


## Train and Test DataFrame

df_test = df_atp_simple[df_atp_simple['Year'].astype('int32')>=2019]
df_train = df_atp_simple[df_atp_simple['Year'].astype('int32')<2019]
#df_train = df_atp_simple
#df_test = df_atp_simple

df_results = pd.DataFrame(columns=['Model',
            'Type',
            'Strategy',
            'Total Earning',
            'Nb Bets',
            'Earned By Bet',
            'Earned on Fav',
            'Loss on Fav',
            'Earned on UnderD',
            'Loss on UnderD'])
## Odd Model 
odd_model = models.Odd_Model(df_train,df_test,df_atp_simple)
#odd_model.modelPlotTest()
odd_model.modelTrain()
df_results = df_results.append(odd_model.modelTestVictories('all_victories'),ignore_index=True)
df_results = df_results.append(odd_model.modelTestVictories('all_predicted'),ignore_index=True)

## Linear Model
linear_model = models.LinearRegression_Model(df_train,df_test,df_atp_simple)
#linear_model.modelTrain()
#linear_model.modelPlotTrain()
#linear_model.modelPlotTest()
#linear_model.modelTestVictories()


## Logistic Model
logistic_model = models.LogisiticRegression_Model(df_train,df_test,df_atp_simple)
#logistic_model.modelTrain()
#logistic_model.modelPlotBookies()
#linear_model.modelPlotTrain()
#linear_model.modelPlotTest()
#linear_model.modelTestVictories()


## Polynomial Model
polynomial_model = models.PolynomialRegression_Model(df_train,df_test,df_atp_simple,4)
polynomial_model.modelTrain()
#polynomial_model.modelPlotBookies()
#polynomial_model.modelPlotTrain()
#polynomial_model.modelPlotTest()
df_results = df_results.append(polynomial_model.modelTestVictories("all_miss_estimate"),ignore_index=True)
df_results = df_results.append(polynomial_model.modelTestVictories("0.1_miss_estimate"),ignore_index=True)
df_results = df_results.append(polynomial_model.modelTestVictories("selfmade_when_bellow"),ignore_index=True)

print(df_results.sort_values(by = ['Earned By Bet'], ascending = False))