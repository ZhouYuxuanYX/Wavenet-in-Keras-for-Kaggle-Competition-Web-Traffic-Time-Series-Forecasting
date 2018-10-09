from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, Concatenate
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import preprocessing
import datetime
import pandas as pd
import pathlib
import gc

def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]

def transform_series_encode(series_array):
    series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1, 1)
    series_std = series_array.std(axis=1).reshape(-1,1)
    epsilon = 1e-6
    series_array = (series_array - series_mean)/(series_std+epsilon)
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))
    return series_array, series_mean, series_std

def untransform_series_decode(series_array,encode_series_mean, encdoe_series_std):
    series_array = series_array.reshape(series_array.shape[0], series_array.shape[1])
    series_array = series_array*encdoe_series_std + encode_series_mean
    # unlog the data, clip the negative part if smaller than 0
    np.clip(np.power(10., series_array) - 1.0, 0.0, None)
    return  series_array

def transform_series_decode(series_array, encode_series_mean, encode_series_std):
    series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
    epsilon = 1e-6 # prevent numerical error in the case std = 0
    series_array = (series_array - encode_series_mean)/(encode_series_std+epsilon)
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))
    return series_array

def predict_sequences(input_sequences, batch_size):
    history_sequences = input_sequences.copy()
    print(history_sequences.shape)
    pred_sequences = np.zeros((history_sequences.shape[0], pred_steps, 1))  # initialize output (pred_steps time steps)
    print(pred_sequences.shape)
    for i in range(pred_steps):
        # record next time step prediction (last time step of model output)
        last_step_pred = model.predict(history_sequences,batch_size)[:, -1, 0]
        print("last step prediction first 10 channels")
        print(last_step_pred[0:10])
        print(last_step_pred.shape)
        pred_sequences[:, i, 0] = last_step_pred

        # add the next time step prediction to the history sequence
        history_sequences = np.concatenate([history_sequences,
                                           last_step_pred.reshape(-1, 1, 1)], axis=1)

    return pred_sequences

def predict_and_plot(encoder_input_data, sample_ind, batch_size, enc_tail_len=50, decoder_target_data=1):
    encode_series = encoder_input_data[sample_ind:sample_ind + 1, :, :]
    pred_series = predict_sequences(encode_series, batch_size)

    encode_series = encode_series.reshape(-1, 1)
    pred_series = pred_series.reshape(-1, 1)

    if isinstance(decoder_target_data, np.ndarray ):
        target_series = decoder_target_data[sample_ind, :, :1].reshape(-1, 1)
        encode_series_tail = np.concatenate([encode_series[-enc_tail_len:], target_series[:1]])
    else:
        encode_series_tail = encode_series[-enc_tail_len:]

    x_encode = encode_series_tail.shape[0]

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, x_encode + 1), encode_series_tail)

    plt.plot(range(x_encode, x_encode + pred_steps), pred_series, color='teal', linestyle='--')

    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)

    if isinstance(decoder_target_data, np.ndarray):
        plt.plot(range(x_encode, x_encode + pred_steps), target_series, color='orange')
        plt.legend(['Encoding Series', 'Target Series', 'Predictions'])
    else:
        plt.legend(['Encoding Series', 'Predictions'])

model_name = 'Wavenet'

# load existing model
load_previous_models = True
if load_previous_models:
    print('Load Previous Models')
    model = load_model(model_name+'.h5')


pages, cmp_enc_start, cmp_enc_end, pred_steps, series_array, data_start_date, data_end_date, train_pred_start, train_pred_end, train_enc_start, \
train_enc_end,  val_enc_start, val_enc_end, date_to_index, val_pred_start, val_pred_end = preprocessing()

gc.collect()

#### Build neural networks ####
if not load_previous_models:
    # convolutional layer oparameters
    n_filters = 32
    filter_width = 2
    dilation_rates = [2**i for i in range(8)]

    # define an input history series and pass it through a stack of dilated causal convolutions
    history_seq = Input(shape=(None, 1))
    x = history_seq

    for dilation_rate in dilation_rates:
        x = Conv1D(filters = n_filters,
                   kernel_size=filter_width,
                   padding='causal',
                   dilation_rate=dilation_rate)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(1)(x)

    # extract the last 14 time steps as the training target
    def slice(x, seq_length):
        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length':62})(x)

    model = Model(history_seq, pred_seq_train)

model.summary()

#### Train neural networks ####
if load_previous_models:
    print('Use Previous Model. Not Training')
else:
    batch_size = 2**8
    epochs = 10

    # sample of series from train_enc_start to train_enc_end
    # sample of series from train_enc_start to train_enc_end
    encoder_input_data = get_time_block_series(series_array, date_to_index,
                                               train_enc_start, train_enc_end)
    encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
    np.isnan(encoder_input_data).any()

    # sample of series from train_pred_start to train_pred_end
    decoder_target_data = get_time_block_series(series_array, date_to_index,
                                                train_pred_start, train_pred_end)
    decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)
    np.isnan(decoder_target_data).any()

    # we append a lagged history of the target series to the input data,
    # so that we can train with teacher forcing
    lagged_target_history = decoder_target_data[:,:-1,:1]
    encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

    model.compile(Adam(), loss='mean_absolute_error')
    history = model.fit(encoder_input_data, decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)
    # save the model
    model.save(model_name + '.h5')


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train','Valid'])

    encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)
    encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)

    decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)
    decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)


    predict_and_plot(encoder_input_data, 100, decoder_target_data=decoder_target_data)
    predict_and_plot(encoder_input_data, 70000, decoder_target_data=decoder_target_data)

#### Predict for the Competition ####
gc.collect()
encoder_input_data = get_time_block_series(series_array, date_to_index, cmp_enc_start, cmp_enc_end)
encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)
pred_series = predict_sequences(encoder_input_data, 2**8)

# visualize one sample to check the prediction
predict_and_plot(encoder_input_data, 100, 2**8)
gc.collect()

# reverse the transformation
pred_series_transformed = untransform_series_decode(pred_series, encode_series_mean, encode_series_std)
gc.collect()

cmp_pred_start = datetime.datetime.strptime(data_end_date, "%Y-%m-%d") + datetime.timedelta(1)
cmp_pred_end = cmp_pred_start + datetime.timedelta(62)
cmp_output_date = pd.Index(np.arange(cmp_pred_start, cmp_pred_end, datetime.timedelta(days=1)).astype('str'))

# check the time frame
print('encode_input_first_day:', cmp_enc_start.date())
print('encode_input_last_day:', cmp_enc_end.date())
print('pred_first_day:', cmp_pred_start.date())
print('pred_last_day:', cmp_pred_end.date())

result_df = pd.DataFrame(pred_series_transformed, columns=cmp_output_date)
result_df['Page'] = pages  # Append 'Page' column from input_df
result_df = pd.melt(result_df, id_vars='Page', var_name='date',
                    value_name='Visits')
gc.collect()

#### Output DataFrame ####
key_file = 'key_2.csv'
print('%%% Reading data', key_file, '...', end='', flush=True)
project_path = pathlib.Path(r'C:\Users\zhouyuxuan\Desktop\google kaggle comptition time series tutorial')
output_df = pd.read_csv(project_path / key_file)
print('done!')

# Peak memory usage: additional 2 GB
output_df['date'] = output_df['Page'].apply(lambda a: a[-10:])  # take the last 10 characters from 'Page' as date
output_df['Page'] = output_df['Page'].apply(lambda a: a[:-11])  # remove the last 10 caharacters from 'Page'
output_df.info()
output_df['date'].values[0:62]
output_df = output_df.merge(result_df, on = "Page", how='left')
del result_df
gc.collect()

# Check if there is any null value
output_df.loc[output_df.Visits.isnull(), 'Visits']

output_file_path = project_path / (model_name + '.csv')

print('%%% Writing result to ' + str(output_file_path) + ' ...',
      end = '', flush = True)
output_df[['Id','Visits']].to_csv(output_file_path, index = False, float_format='%.3f')
print('done!')
gc.collect()




