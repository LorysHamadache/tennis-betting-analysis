######## Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
pd.options.mode.chained_assignment = None
from itertools import product



class Model():
    model_name = 'Base Model'

    def modelFunction(self,x):
        return x

    def modelTestOddWinrate(self,df_data):

        plot_data = df_data.groupby(['favorite_odd','favorite_won']).agg({'Name':'count'})

        plot_data = plot_data.reset_index()
        combs = pd.DataFrame(list(product(plot_data['favorite_odd'].unique(), plot_data['favorite_won'].unique())),  columns=['favorite_odd', 'favorite_won'])
        plot_data = plot_data.merge(combs, how = 'right').fillna(0)
        plot_data = plot_data.set_index(['favorite_odd','favorite_won'])
        plot_data = plot_data.groupby(['favorite_odd'], as_index=True).apply(lambda x: x / x.sum())

        plot_data = plot_data.query('favorite_won == 1')
        plot_data = plot_data.rename(columns={"Name":"probability"})
        plot_data = plot_data.reset_index()
        plot_data = plot_data[['favorite_odd','probability']]

        return plot_data
    
    def modelTrain(self):
        self.trained_model = 1
        return

    def __init__(self,df_train,df_test,df_final):
        print("Running Model: ",self.model_name)
        self.trained_model = None
        self.train_data = df_train.copy()
        self.test_data = df_test.copy()
        self.total_data = df_final.copy()
        self.plot_test_data = self.modelTestOddWinrate(self.test_data)
        self.plot_train_data = self.modelTestOddWinrate(self.train_data)

    def modelPlotTest(self):
        x = self.plot_test_data['favorite_odd']
        x_ = pd.Series(np.linspace(1,3.5,100))
        plt.plot(x_,self.modelFunction(x_), label = self.model_name)
        plt.plot(x,self.plot_test_data['probability'], 'x', label = 'Test Data')
        plot_title = "Model: " + self.model_name + "  vs Test Data"
        plt.title(plot_title)
        plt.legend()
        plt.show()
    
    def modelPlotTrain(self):
        x = self.plot_train_data['favorite_odd']
        x_ = pd.Series(np.linspace(1,3.5,100))
        plt.plot(x_,self.modelFunction(x_), label = self.model_name)
        plt.plot(x,self.plot_train_data['probability'], 'x', label = 'Train Data')
        plot_title = "Model: " + self.model_name + "  vs Train Data"
        plt.title(plot_title)
        plt.legend()
        plt.show()

    def modelPlotBookies(self):
        
        x = pd.Series(np.linspace(1,3.5,100))

        x_real = self.plot_train_data['favorite_odd']
        y_real = self.plot_train_data['probability']

        plt.plot(x_real, y_real,'x',label = 'Frequency By Odd')
        plt.plot(x,self.modelFunction(x), label = self.model_name)
        plt.plot(x,x.apply(lambda x: 1/x), label = 'Bookmakers')
        plot_title = "Model: " + self.model_name + "  vs BookMakers"
        plt.title(plot_title)
        plt.legend()
        plt.ylim(0, 1.01)
        plt.show()

    def modelTestVictories(self, strategy):
        if (self.trained_model is None):
            print('Untrained Model')
            return
        elif (strategy == 'all_predicted'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_predict = self.modelFunction(x_test)
            y_predict = pd.Series((y_predict>0.5).astype(int), index = y_test.index)
        elif (strategy == 'all_victories'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_predict = self.modelFunction(x_test)
            y_predict = pd.Series(np.ones(len(x_test)).astype(int), index = y_test.index)
            
        elif (strategy == 'all_miss_estimate'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_model = self.modelFunction(x_test)
            y_odds = x_test.apply(lambda x:1/x)
            y_predict = pd.Series((y_model-y_odds>0).astype(int), index = y_test.index)

        elif (strategy == '0.1_miss_estimate'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_model = self.modelFunction(x_test)
            y_odds = x_test.apply(lambda x:1/x)
            condition  = ((y_model  >= y_odds -0.01) & y_model>0.5)
            y_predict = pd.Series(condition.astype(int), index = y_test.index)
        elif (strategy == 'selfmade_when_bellow'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_model = self.modelFunction(x_test)
            y_odds = x_test.apply(lambda x:1/x)
            condition  = (y_model  >= y_odds -0.05) 
            y_predict = pd.Series(condition.astype(int), index = y_test.index)
            y_predict = y_predict[y_predict == 0]
        elif (strategy == 'selfmade_hardcoded'):
            x_test = self.test_data['favorite_odd']
            y_test = self.test_data['favorite_won']
            y_model = self.modelFunction(x_test)
            y_odds = x_test.apply(lambda x:1/x)
            condition  = (x_test  >= 1.45) 
            y_predict = pd.Series(condition.astype(int), index = y_test.index)
            y_predict = y_predict[y_predict == 1]
        

        self.test_data['Predicted'] = y_predict

        earned_money_favorite = self.test_data.query('favorite_won == 1 & Predicted ==1')['favorite_odd'].sum() - len(self.test_data.query('favorite_won == 1 & Predicted ==1')['favorite_odd'])
        earned_money_underdog = self.test_data.query('favorite_won == 0 & Predicted ==0')['nonfavorite_odd'].sum() - len(self.test_data.query('favorite_won == 0 & Predicted ==0')['nonfavorite_odd'])
        loss_money_underdog= len(self.test_data.query('favorite_won == 1 & Predicted ==0').index)
        loss_money_favorite = len(self.test_data.query('favorite_won == 0 & Predicted ==1').index)

        total = earned_money_favorite + earned_money_underdog - loss_money_favorite -loss_money_underdog

        results = {
            'Model': self.model_name,
            'Type': "1$ Bets",
            'Strategy': strategy,
            'Total Earning': round(total),
            'Nb Bets': len(y_predict),
            'Earned By Bet': total/len(y_predict),
            'Earned on Fav':earned_money_favorite,
            'Loss on Fav': loss_money_favorite,
            'Earned on UnderD': earned_money_underdog,
            'Loss on UnderD': loss_money_underdog
        }
        return results


class Odd_Model(Model):

    model_name = 'Odd Model'
    
    def modelFunction(self,x): # favorite odds
        return  1/(x) #probability of favorite wwinning



class LinearRegression_Model(Model):

    model_name = 'Linear Regression Model'
    def modelFunction(self,x): # favorite odds
        if (self.trained_model is None):
            return
        else:
            results = self.trained_model.predict(x.to_numpy().reshape(-1, 1))
            return results
    
    
    def  modelTrain(self):
        
        y_train = self.train_data['favorite_won'].to_numpy()
        x_train = self.train_data['favorite_odd'].to_numpy()
        self.trained_model = linear_model.LinearRegression()
        self.trained_model.fit(x_train.reshape(-1, 1), y_train)



class LogisiticRegression_Model(Model):

    model_name = 'Logistic Regression Model'

    def modelFunction(self,x): # favorite odds
        if (self.trained_model is None):
            return
        else:
            x = (x - self.total_data['favorite_odd'].min())/ (self.total_data['favorite_odd'].max()- self.total_data['favorite_odd'].min())
            results = self.trained_model.predict_proba(x.to_numpy().reshape(-1, 1))
            return results[:,1]
    
    
    def  modelTrain(self):
        
        y_train = self.train_data['favorite_won'].to_numpy()
        x_train = self.train_data['favorite_odd_normalized'].to_numpy()
        self.trained_model = linear_model.LogisticRegression()
        self.trained_model.fit(x_train.reshape(-1, 1), y_train)



class PolynomialRegression_Model(Model):

    model_name = 'Polynomial Regression Model'

    def __init__(self,df_train,df_test,df_final, degree):
        Model.__init__(self,df_train,df_test,df_final)
        self.degree = degree


    def modelFunction(self,x): # favorite odds
        if (self.trained_model is None):
            return
        else:
            polyn = preprocessing.PolynomialFeatures(degree= self.degree)
            x_ = polyn.fit_transform(x.to_numpy().reshape(-1, 1))

            results = self.trained_model.predict(x_)
            return results
    
    
    def  modelTrain(self):
        
        y_train = self.train_data['favorite_won'].to_numpy()
        x_train = self.train_data['favorite_odd'].to_numpy()


        polyn = preprocessing.PolynomialFeatures(degree= self.degree)
        x_train_degree = polyn.fit_transform(x_train.reshape(-1, 1))

        self.trained_model = linear_model.LinearRegression()
        self.trained_model.fit(x_train_degree, y_train)
