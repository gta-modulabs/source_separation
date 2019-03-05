import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import ntpath

def get_data_files(data_dir):
    data_files = [join(data_dir, f) for f in listdir(data_dir) 
                        if isfile(join(data_dir, f)) and '.npz' in f]
    return data_files

def get_data(batch, length, data_files):
    
    '''
    #parameters
    
    batch : scalar, the size of batch
    length : scalar, the length of batch
    data_files : string, the file pathes to load
    
    #returns
    
    audios : array (batch, length), mixed waves
    vocals : array (batch, length), separated vocal waves
    '''

    read_length = 0
    audios = []
    vocals = []
    
    while(read_length < batch * length):
        file_index = np.random.randint(0, len(data_files), 1)[0]
        data = np.load(data_files[file_index])
        
        audio = data['audio']
        vocal = data['vocals']
        
        need_length = batch * length - read_length
        
        if audio.shape[0] > need_length:
            audio = audio[:need_length]
            vocal = vocal[:need_length]
            
        read_length += audio.shape[0]
        audios.append(audio)
        vocals.append(vocal)
        
    audios = np.concatenate(audios, axis=0)
    audios = np.reshape(audios, [batch, length, 2])
    
    vocals = np.concatenate(vocals, axis=0)
    vocals = np.reshape(vocals, [batch, length, 2])
    
    return audios, vocals
    
def stft(x, fft_size, hop_size):
    '''
    parameter
    x : [Batch, Length, Channels=2]
    
    return value
    y_magnitude : [Batch, FrameLength, Bins=1025, Channels=2]
    y_phase : [Batch, FrameLength, Bins=1025, Channels=2]
    '''
        
    # [Batch, Channels, Length]
    x_tranposed = tf.transpose(x, [0, 2, 1])
    
    # [Batch, Channels, FrameLength, Bins]
    x_stft = tf.contrib.signal.stft(signals=x_tranposed, 
                                    frame_length=fft_size, 
                                    frame_step=hop_size, pad_end=True)
    
    # [Batch, FrameLength, Bins, Channels]
    y_complex = tf.transpose(x_stft, [0, 2, 3, 1])
    
    y_magnitude = tf.abs(y_complex)
    y_phase = y_complex / tf.cast(y_magnitude, tf.complex64)
    
    y_magnitude = tf.log(y_magnitude + 1)
    
    return y_magnitude, y_phase

def reconstruct(source_magnitude, source_phase, fft_size, hop_size, griffin_lim_iters=40):
    '''
    parameter
    source_magnitude : [Batch, FrameLength, Bins, Channels]
    source_phase : [Batch, FrameLength, Bins, Channels]
    '''
    
    # [Batch, Channels, FrameLength, Bins]
    source_magnitude = tf.transpose(source_magnitude, [0, 3, 1, 2])
    source_phase = tf.transpose(source_phase, [0, 3, 1, 2])
    
    source_magnitude = tf.exp(source_magnitude) - 1
    source_complex = tf.cast(source_magnitude, tf.complex64) * source_phase
    
    # Griffin-Lim
    for _ in range(griffin_lim_iters):
        # [Batch, Channels, Length]
        reconstructed_wave = tf.contrib.signal.inverse_stft(stfts=source_complex, 
                                       frame_length=fft_size, frame_step=hop_size)
                                       
        # [Batch, Channels, FrameLength, Bins]
        source_complex = tf.contrib.signal.stft(signals=reconstructed_wave, 
                                    frame_length=fft_size, 
                                    frame_step=hop_size)
        source_magnitude_temp = tf.abs(source_complex)
        source_phase = source_complex / tf.cast(source_magnitude_temp, tf.complex64)
        source_complex = tf.cast(source_magnitude, tf.complex64) * source_phase
        
    # [Batch, Length, Channels]
    reconstructed_wave = tf.transpose(reconstructed_wave, [0, 2, 1])
    
    return reconstructed_wave
    
def safe_log(x, eps=1e-5):
    return np.log(np.maximum(x, eps)) 
    