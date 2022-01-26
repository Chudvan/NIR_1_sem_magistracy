import pandas as pd
import re
import numpy as np
from itertools import chain
from NeuralNetwork.tools import groupby


class NeuralNetwork:
    def __init__(self, x, y, model=None):
        self.df_x = x
        for field in self.df_x.columns:
            self.df_x[field] = self.df_x[field].apply(lambda entry: float(entry))
        self.df_y = y
        for field in self.df_y.columns:
            self.df_y[field] = self.df_y[field].apply(lambda entry: float(entry))
        self.model = model
        self.from_ = 0
        self.to = 1
    
    def get_df_xy(self, from_ = None, to = None, xy = None):
        if xy is None:
            xy = 'x'
        
        if from_ is None:
            from_ = self.from_
        else:
            self.from_ = from_
            
        if to is None:
            to = self.to
        else:
            self.to = to
        
        if xy.lower() == 'x':   
            return self.df_x[from_:to]
        elif xy.lower() == 'y':   
            return self.df_y[from_:to]
    
    def predict(self, test=None):
        if self.model is None:
            raise Exception('You can use predict method only with model.')
    
        if test is None:
            df_x = self.get_df_xy(self.from_, self.to)
            df_res = pd.DataFrame(self.model.predict(df_x.values))
            df_res.columns = self.df_y.columns
            df_res.index = df_x.index
        else:
            assert all([column == self.df_y.columns[i] # ['Neutral', 'Happy', 'Sad', 'Angry',
                        for i, column in enumerate(test.columns[:7])]) # 'Surprised', 'Scared', 'Disgusted']
            df_res = pd.DataFrame(self.model.predict(test[['Valence', 'Arousal']].values))
            df_res.columns = self.df_y.columns
            df_res.index = test.index
        return df_res
    
    def get_test(self, n=None):
        if n is None:
            test = pd.concat([self.df_y, self.df_x], axis=1)
        else:
            test = pd.concat([self.df_y, self.df_x], axis=1).iloc[n:n+1]
        return test
    
    def get_diff(self, test):
        predict_df = self.predict(test)
        predict_values = predict_df.values
        predict_values -= test[test.columns[:7]].values
        diff_df = pd.DataFrame(predict_values)
        diff_df.columns = predict_df.columns
        diff_df.index = predict_df.index
        return diff_df
    
    def create_add_to_index(self, csv_file):   
        res = []
        without_participant = re.split('Participant \d*', csv_file)[1]
        fragments = re.split('Analysis ', without_participant)
        res.append(fragments[0])
        res.append(re.split('_video_', fragments[1])[0])
        return ''.join(res)
    
    def model_metric(self, test, type_='mean'):        
        if type_ == 'mean':
            array = np.absolute(self.get_diff(test).values)
            coefs = np.array(range(array.shape[1] + 1))[1:]
            for i in range(array.shape[0]):
                array[i].sort()
                array[i] *= coefs
            return np.sum(array) / (array.shape[0] * np.sum(coefs))
        elif type_ == 'norm':
            array = self.get_diff(test).values
            sum_ = 0
            for vector in array:
                sum_ += np.linalg.norm(vector)
            return sum_ / array.shape[0]
        elif type_ == 'stat':
            stat = self.statistics(test)
            vector = np.absolute(stat.values)
            return vector.mean()
        else:
            raise Exception('Unknown metric')
            
    def statistics(self, test):        
        diff = self.get_diff(test)
        columns = [['min_' + emotion, 'max_' + emotion, 'mean_abs_' + emotion]
                   for emotion in diff.columns]
        columns = list(chain.from_iterable(columns))
        statistics_df = pd.DataFrame(columns=columns)
        entry_dict = {}
        for emotion in diff.columns:
            entry_dict['min_' + emotion] = np.min(diff[emotion])
            entry_dict['max_' + emotion] = np.max(diff[emotion])
            entry_dict['mean_abs_' + emotion] = np.mean(np.absolute(diff[emotion]))
        statistics_df = statistics_df.append(entry_dict, ignore_index = True)
        return statistics_df

    def create_train_df_from_diff(self, test):
        pa_fields = list(self.df_x.columns)
        seven_fields = list(self.df_y.columns)
        
        index_df = pd.DataFrame(columns=['Index_'])
        
        diff = self.get_diff(test)
        
        index_set = set()
        for emotion in seven_fields:
            diff_cur_emotion = diff[np.absolute(diff[emotion]) > 0.1]
            index_set = index_set.union(set(diff_cur_emotion.index))
        index_df = pd.DataFrame({'Index_': list(index_set)})
        index_df.index = index_df['Index_']
        
        train_df = pd.merge(test, index_df, left_index=True, right_index=True)[seven_fields + pa_fields]
        train_df.insert(0, 'Index_', train_df.index)
        return groupby(train_df)

