import pandas as pd


class NeuralNetwork:
    def __init__(self, x, y, model):
        self.df_x = x
        self.df_y = y
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
    
    def predict(self):
        df_x = self.get_df_xy(self.from_, self.to)
        df_res = pd.DataFrame(self.model.predict(df_x.values))
        df_res.columns = self.df_y.columns
        df_res.index = df_x.index
        return df_res

