import os
import sqlite3
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display_html
from itertools import chain,cycle


pa_fields =     [
    'Valence',
    'Arousal'
]

seven_fields = [
    'Neutral', 
    'Happy', 
    'Sad', 
    'Angry', 
    'Surprised', 
    'Scared', 
    'Disgusted'
]

fields = seven_fields + pa_fields

metrics = ['mean', 'norm', 'stat']

clear_count_dict = {
    'Neutral': 200,
    'Happy': 200,
    'Sad': 14,
    'Angry': 44,
    'Surprised': 30,
    'Scared': 12,
    'Disgusted': 30
}


def replace_end_symb(csv_path, encoding=None):
    if not encoding:
        encoding = "UTF-8"
    f = open(csv_path, encoding = encoding)
    text = f.read()
    text = text.replace('\t\n', '\n')
    f.close()
    f = open(csv_path, 'w', encoding = encoding)
    f.write(text)
    f.close()
    
def create_correct_df(data_dir, csv_file, encoding=None):
    from .neural_network import NeuralNetwork
    if not encoding:
        encoding = "UTF-8"
    csv_path = os.path.join(data_dir, csv_file)
    replace_end_symb(csv_path, encoding)
    df = pd.read_csv(csv_path, sep='\t', skiprows = lambda i: i in list(range(8)), encoding = encoding)
    if len(df.columns) < 70:
        df = pd.read_csv(csv_path, sep=';', skiprows = lambda i: i in list(range(8)), encoding = encoding)
    df_first = df.columns[0]
    unknown_index = list(df.columns).index('Event Marker')
    df.columns = df.columns[1:].insert(unknown_index, 'UNKNOWN')
    df.insert(0, df_first, df.index)
    add_to_index = NeuralNetwork.create_add_to_index(csv_file)
    df.index = [index + add_to_index for index in df.index]
    df.insert(0, 'Index_', df.index)
    df = df[ # Without FIT_FAILED and FIND_FAILED
        (df['Neutral'] != 'FIT_FAILED') & 
        (df['Neutral'] != 'FIND_FAILED')
    ]
    return df
    
def concat_next_csv(df, data_dir, csv_file, encoding=None):
    df_2 = create_correct_df(data_dir, csv_file, encoding=None)
    df = pd.concat([df, df_2], axis=0)
    return df

def save_to_db(db_path, name_db, df):
    connection = sqlite3.connect(db_path)
    df_columns = [field.replace('-', '_') for field in df.columns]
    df_columns = [field.replace(' ', '_') for field in df_columns]
    try:
        i = df_columns.index('3d_Landmarks')
        df_columns[i] = 'three_d_Landmarks'
    except ValueError:
        pass
    fields = ',\n'.join([f'\t{field} TEXT' for field in df_columns])
    create_costs_table_query = f"""
create table {name_db} (
{fields}
)
"""
    connection.execute(create_costs_table_query)
    connection.commit()
    values = ', '.join(['?' for _ in range(len(df.columns))])
    for row in df.iterrows():
        connection.execute(f"INSERT OR IGNORE INTO {name_db} VALUES({values})", tuple(row[1]))
    connection.commit()
    return connection
    
def load_from_db(db_path, name_db):
    connection = sqlite3.connect(db_path)
    df = pd.read_sql(f'select * from {name_db}', con=connection)
    return df

def groupby(df, by=None, prediction=2, other=False, other_groupby=True):
    if by is None:
        by = pa_fields
        
    df_copy = df[seven_fields + pa_fields].copy()
    
    for field in pa_fields:
        df_copy[field] = df_copy[field].apply(lambda x: round(float(x), prediction))
    for field in seven_fields:
        df_copy[field] = df_copy[field].apply(lambda x: float(x))
    
    df_copy.index = df['Index_']
    
    groupby_fields_sorted = list(sorted(df_copy.groupby(by), key=lambda x: -len(x[1])))
    for group in groupby_fields_sorted:
        for field in seven_fields:
            group[1][field] = round(group[1][field].mean(), prediction)
            
    df_train = pd.DataFrame()
    if other:
        df_other = pd.DataFrame()
    
    for group in groupby_fields_sorted:
        len_group = len(group[1])
        ln_ = np.log10(len_group)
        rand_set = set()
        for _ in range(int(round(ln_, 0)) + 1):
            i = random.randint(0, len_group - 1)
            while i in rand_set:
                i = random.randint(0, len_group - 1)
            rand_set.add(i)
            df_train = pd.concat([df_train, group[1].iloc[i:i + 1]], axis=0)
        if other:
            all_i_without_rand_set = set(range(len_group)) - rand_set
            if other_groupby:
            	df_other = pd.concat([df_other, group[1].iloc[list(all_i_without_rand_set)]], axis=0)
            else:
            	df_other = pd.concat([df_other, df.iloc[list(all_i_without_rand_set)]], axis=0)
    if other:
        for field in seven_fields + pa_fields:
            df_other[field] = df_other[field].apply(lambda x: float(x))
        return df_train, df_other[seven_fields + pa_fields]
    return df_train

def apply_float(df_, columns):
    for field in columns:
        df_[field] = df_[field].apply(lambda el: float(el))
        
def make_valid_df(df_, columns=None):
    if columns is not None:
        apply_float(df_, columns)
    df_.index = df_['Index_']
    
def refitting(models, test, df_metrics, df_train=None, v=1, 
              layer='first', epochs=20, batch_size=20, type_='diff'):
    for nn_list in models:
        nn_list[0] = nn_list[0].split('_')[0] + f'_{v}'
        nn = nn_list[2]
        print('refit', nn_list[0])
        if type_ == 'diff':
            df_train = nn.create_train_df_from_diff(test)
        elif type_ == 'split' and df_train is not None:
            pass
        else:
            raise Exception('Unknown refitting type.')
        nn.fit(df_train, epochs=epochs, batch_size=batch_size)
        entry_dict = {'model': nn_list[0], 'layer': layer, 'N': nn_list[1]}
        entry_dict.update({metric: nn.model_metric(test, metric) for metric in metrics})
        df_metrics = df_metrics.append(entry_dict, ignore_index = True)
        print(entry_dict)
    return df_metrics

def plot_emotions(models, df_clear, df_clear_metrics, fitting_algorithm='another', df_metrics=None, scale=False, figsize=(20, 15)):
    plt.figure(figsize=figsize)
    if fitting_algorithm == 'another':
        for i, model_tuple in enumerate(models):
            entry_dict = {'model': model_tuple[0]}
            nn = model_tuple[2]
            clear_metric, emotion_mean_values = nn.model_metric(df_clear, 'clear', scale=scale)
            entry_dict.update({'clear': clear_metric})
            for j, emotion in enumerate(df_clear.columns[:7]):
                entry_dict.update({emotion: emotion_mean_values[j]})

            plt.plot(seven_fields, emotion_mean_values, label=model_tuple[0])
            # entry_dict.update({metric: df_metrics.iloc[i][metric] for metric in metrics})
            df_clear_metrics = df_clear_metrics.append(entry_dict, ignore_index = True)
        plt.xlabel("Эмоции")
        plt.ylabel("Средние значения предсказанных чистых эмоций / Средние значения чистых эмоций")
    else:
        fitting_algorithm = 'master'
        for i, model_tuple in enumerate(models):
            values = model_tuple[2].predict(df_clear).max().values
            if scale:
                values /= df_clear.max().values[:-2]
            plt.plot(seven_fields, values, label=model_tuple[0])
            entry_dict = {'model': model_tuple[0]}
            entry_dict.update({metric: df_metrics.iloc[i][metric] for metric in metrics})
            entry_dict.update({emotion: values[j] for j, emotion in enumerate(seven_fields)})
            df_clear_metrics = df_clear_metrics.append(entry_dict, ignore_index = True)
        plt.xlabel("Эмоции")
        plt.ylabel("Максимальные значения")
    plt.legend()
    plt.show()
    return df_clear_metrics

def create_metric_df_dict(metrics, df_metrics, df_clear_metrics):
    metric_df_dict = {metric: df_metrics for metric in metrics[:-1]}
    metric_df_dict.update({metrics[-1]: df_clear_metrics})
    return metric_df_dict

def plot_metrics(metric_df_dict, layer='first'):
    # dependencies
    mean_ = 'mean'
    clear = 'clear'
    
    x = []
    y = []
    
    df_metrics = metric_df_dict[mean_]
    metrics = list(metric_df_dict.keys())
    
    for metric in metrics:
        if layer == 'first':
            x.append(df_metrics['N'])
        else:
            x.append(df_metrics.index)
        df_ = metric_df_dict[metric]
        y.append(df_[metric])
    
    for i in range(len(metrics)):
        plt.plot(x[i], y[i], label=metrics[i])
        plt.xlabel("Число нейронов N в 1 скрытом слое")
        if metrics[i] == clear:
            plt.ylabel("Сумма средних значений предсказанных чистых эмоций / Сумму средних значений чистых эмоций")
        else:
            plt.ylabel("Ошибка")
        plt.legend()
        plt.show()

def save_models(models, path_to_saved_models, layer='first', v=1):
    dir_path = os.path.join(path_to_saved_models, layer, f'_{v}')
    for model_list in models:
        N = model_list[1]
        nn = model_list[2]
        save_name = f'model_{layer}_{N}_{v}'
        path = os.path.join(dir_path, save_name)
        nn.model.save(path)

def _removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def load_models(path_to_saved_models, df, models_list=None, layer='first', 
                v=1, sort=True, csv_test_file=None):
    from tensorflow.keras.models import load_model
    from nn_train.neural_network import NeuralNetwork

    dir_path = os.path.join(path_to_saved_models, layer, f'_{v}')
    if models_list is None:
        models = list(os.walk(dir_path))[0][1]
    else:
        prefix = f'model_{layer}_'
        if (layer == 'first' or layer == 'third') and v == 1:
            models = [prefix + N for N in models_list]
        elif layer == 'third' and v != 2:
            models = [N + f'_{v}' for N in models_list]
        else:
            models = [prefix + N + f'_{v}' for N in models_list]
#     print(models)
    for i in range(len(models)):
        model_layers_v = _removeprefix(models[i], f'model_{layer}_')
        N = model_layers_v.split('_')[0]
        path = os.path.join(dir_path, models[i])
        model = load_model(path)
        nn = NeuralNetwork(df[pa_fields], df[seven_fields], model, csv_test_file)
        models[i] = [model_layers_v, N, nn]
    
    if sort:
    	models.sort(key=lambda x: list(map(int, x[1].split('.'))))
    
    return models

def create_df_metrics(models, test, df_metrics, layer='first'):
    for model_list in models:
        entry_dict = {'model': model_list[0], 'layer': layer, 'N': model_list[1]}
        entry_dict.update({metric: model_list[2].model_metric(test, metric) for metric in metrics})
        df_metrics = df_metrics.append(entry_dict, ignore_index = True)
    return df_metrics
    
def display_dfs(*args, titles=cycle(['']), mode='column'):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>']))):
        cur_html_str = '<td style="vertical-align:top">'
        if mode == 'column':
            cur_html_str+=f'<h2 align="left">{title}</h2>'
        elif mode == 'row':
            cur_html_str+=f'<h2 align="center">{title}</h2>'
        else:
            raise Exception(f'Unknown mode: {mode}')
        cur_html_str+=df.to_html().replace('table','table style="display:inline" align="left"')
        cur_html_str+='</td>'
        if mode == 'column':
            cur_html_str = '<tr align="left">' + cur_html_str + '</tr>'
        elif mode == 'row':
            pass
        else:
            raise Exception(f'Unknown mode: {mode}')
        html_str += cur_html_str
    display_html(html_str,raw=True)
