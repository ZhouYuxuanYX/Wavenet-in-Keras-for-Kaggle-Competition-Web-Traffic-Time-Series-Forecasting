import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from datetime import timedelta

def preprocessing():

    sns.set()
    project_path = pathlib.Path(r'C:\Users\zhouyuxuan\Desktop\google kaggle comptition time series tutorial')
    df = pd.read_csv(project_path/'train_2.csv')
    df.head()
    data_start_date = df.columns[1]
    data_end_date = df.columns[-1]
    print('Data ranges from %s to %s' % (data_start_date, data_end_date))


    def plot_random_series(df, n_series):
        sample = df.sample(n_series, random_state=8)
        page_labels = sample['Page'].tolist()
        series_samples = sample.loc[:, data_start_date:data_end_date]

        plt.figure(figsize=(10, 6))

        for i in range(series_samples.shape[0]):
            np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)

        plt.title('Randomly Selected Wikipedia Page Daily Views Over Time (Log(views) + 1)')
        plt.legend(page_labels)


    plot_random_series(df, 6)
    pred_steps = 62
    pred_length = timedelta(pred_steps)

    first_day = pd.to_datetime(data_start_date)
    last_day = pd.to_datetime(data_end_date)

    val_pred_start = last_day - pred_length + timedelta(1)
    val_pred_end = last_day

    train_pred_start = val_pred_start - pred_length
    train_pred_end = val_pred_start - timedelta(days=1)
    enc_length = train_pred_start - first_day

    train_enc_start = first_day
    train_enc_end = train_enc_start + enc_length - timedelta(1)

    val_enc_start = train_enc_start + pred_length
    val_enc_end = val_enc_start + enc_length - timedelta(1)

    cmp_enc_start = last_day-timedelta(days=255)
    cmp_enc_end = last_day

    print('Train encoding:', train_enc_start, '-', train_enc_end)
    print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')
    print('Val encoding:', val_enc_start, '-', val_enc_end)
    print('Val prediction:', val_pred_start, '-', val_pred_end)

    print('\nEncoding interval:', enc_length.days)
    print('Prediction interval:', pred_length.days)

    #### Input data formatting

    date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                              data=[i for i in range(len(df.columns[1:]))])

    series_array = df[df.columns[1:]].values
    pages = df['Page']
    return pages, cmp_enc_start, cmp_enc_end, pred_steps, series_array, data_start_date, data_end_date, train_pred_start, train_pred_end, train_enc_start,\
           train_enc_end,  val_enc_start, val_enc_end, date_to_index, val_pred_start, val_pred_end







