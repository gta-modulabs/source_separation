import tensorflow as tf

def batch_norm(x, training):
    return tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-5, training=training)

def composite_layer(x, growth_rate, training):
    
    # batch norm
    y = batch_norm(x, training=training)
    # relu
    y = tf.nn.relu(y)
    # conv
    y = tf.layers.conv2d(y, filters=growth_rate, kernel_size=[3, 3], padding='same')
    
    return y

def bottleneck_layer(x, growth_rate, training):
    
    # batch norm
    y = batch_norm(x, training=training)
    # relu
    y = tf.nn.relu(y)
    # conv
    y = tf.layers.conv2d(y, filters=growth_rate * 4, kernel_size=[1, 1], padding='same')
    
    return y

def down_sample(x, k):
    #conv
    y = tf.layers.conv2d(x, filters=k, kernel_size=[1, 1], padding='same')
    #pooling
    y = tf.layers.average_pooling2d(y, pool_size=[2, 2], strides=[2, 2])
        
    return y

def up_sample(x, k):
    # transposed_conv
    y = tf.layers.conv2d_transpose(x, filters=k, kernel_size=[2, 2], strides=[2, 2], padding='same')
    return y
    
def dense_block(x, k, L, training):
    
    y_composite = x
    
    for _ in range(L):    
        y_bottle = bottleneck_layer(y_composite, k, training)
        y_composite_new = composite_layer(y_bottle, k, training)
        y_composite = tf.concat([y_composite, y_composite_new], axis=-1)
        
    return y_composite_new

def LSTM_layer(x, hidden_units):
    # parameters
    # x : [Batch, FrameAxis, FreqAxis, Channels]
    
    # y : [Batch, FrameAxis, FreqAxis]
    y = tf.layers.conv2d(x, filters=1, kernel_size=[1, 1], padding='same')
    y = tf.squeeze(y, axis=3)
    
    lstm = tf.keras.layers.CuDNNLSTM(units=hidden_units, return_sequences=True)
    y = lstm(y)
    
    # y : [Batch, FrameAxis, FreqAxis]
    y = tf.layers.dense(y, x.shape[-2])
    
    # y : [Batch, FrameAxis, FreqAxis, 1]
    return tf.expand_dims(y, axis=-1)

def densenet_band1(x, name='densenet_band1', training=True, reuse=False):
    
    # default params
    growth_rate = 14
    dense_block_nums = 5
    
    with tf.variable_scope(name, reuse=reuse):
        # d1
        d1 = tf.layers.conv2d(x, filters=growth_rate, kernel_size=[3, 3], padding='same')
        d1 = dense_block(d1, growth_rate, dense_block_nums, training)
        ##print(d1)
        
        # d2 
        d2 = down_sample(d1, growth_rate)
        d2 = dense_block(d2, growth_rate, dense_block_nums, training)
        ##print(d2)

        # d3
        d3 = down_sample(d2, growth_rate)
        d3 = dense_block(d3, growth_rate, dense_block_nums, training)
        ##print(d3)

        # d4
        d4 = down_sample(d3, growth_rate)
        d4 = dense_block(d4, growth_rate, dense_block_nums, training)
        d4_lstm = LSTM_layer(d4, 128)
        d4 = tf.concat([d4, d4_lstm], axis=-1)
        ##print(d4)
        
        # u3
        u3 = up_sample(d4, growth_rate)
        u3 = dense_block(u3, growth_rate, dense_block_nums, training)
        u3 = tf.concat([u3, d3], axis=-1)
        ##print(u3)

        # u2
        u2 = up_sample(u3, growth_rate)
        u2 = dense_block(u2, growth_rate, dense_block_nums, training)
        u2 = tf.concat([u2, d2], axis=-1)
        u2_lstm = LSTM_layer(u2, 128)
        u2 = tf.concat([u2, u2_lstm], axis=-1)
        #print(u3)

        # u1
        u1 = up_sample(u2, growth_rate)
        u1 = dense_block(u1, growth_rate, dense_block_nums, training)
        u1 = tf.concat([u1, d1], axis=-1)
        #print(u1)

        return dense_block(u1, 12, 3, training)
    
def densenet_band2(x, name='densenet_band2', training=True, reuse=False):
    
    # default params
    growth_rate = 4
    dense_block_nums = 4
    
    with tf.variable_scope(name, reuse=reuse):
        # d1
        d1 = tf.layers.conv2d(x, filters=growth_rate, kernel_size=[3, 3], padding='same')
        d1 = dense_block(d1, growth_rate, dense_block_nums, training)
        #print(d1)
        
        # d2 
        d2 = down_sample(d1, growth_rate)
        d2 = dense_block(d2, growth_rate, dense_block_nums, training)
        #print(d2)

        # d3
        d3 = down_sample(d2, growth_rate)
        d3 = dense_block(d3, growth_rate, dense_block_nums, training)
        #print(d3)

        # d4
        d4 = down_sample(d3, growth_rate)
        d4 = dense_block(d4, growth_rate, dense_block_nums, training)
        d4_lstm = LSTM_layer(d4, 32)
        d4 = tf.concat([d4, d4_lstm], axis=-1)
        #print(d4)
        
        # u3
        u3 = up_sample(d4, growth_rate)
        u3 = dense_block(u3, growth_rate, dense_block_nums, training)
        u3 = tf.concat([u3, d3], axis=-1)
        #print(u3)

        # u2
        u2 = up_sample(u3, growth_rate)
        u2 = dense_block(u2, growth_rate, dense_block_nums, training)
        u2 = tf.concat([u2, d2], axis=-1)
        #print(u3)

        # u1
        u1 = up_sample(u2, growth_rate)
        u1 = dense_block(u1, growth_rate, dense_block_nums, training)
        u1 = tf.concat([u1, d1], axis=-1)
        #print(u1)

        return dense_block(u1, 12, 3, training)
        
def densenet_band3(x, name='densenet_band3', training=True, reuse=False):
    
    # default params
    growth_rate = 2
    dense_block_nums = 1
    
    with tf.variable_scope(name, reuse=reuse):
        # d1
        d1 = tf.layers.conv2d(x, filters=growth_rate, kernel_size=[3, 3], padding='same')
        d1 = dense_block(d1, growth_rate, dense_block_nums, training)
        #print(d1)
        
        # d2 
        d2 = down_sample(d1, growth_rate)
        d2 = dense_block(d2, growth_rate, dense_block_nums, training)
        #print(d2)

        # d3 
        d3 = down_sample(d2, growth_rate)
        d3 = dense_block(d3, growth_rate, dense_block_nums, training)
        d3_lstm = LSTM_layer(d3, 8)
        d3 = tf.concat([d3, d3_lstm], axis=-1)
        #print(d3)

        # u2
        u2 = up_sample(d3, growth_rate)
        u2 = dense_block(u2, growth_rate, dense_block_nums, training)
        u2 = tf.concat([u2, d2], axis=-1)
        #print(u2)

        # u1
        u1 = up_sample(u2, growth_rate)
        u1 = dense_block(u1, growth_rate, dense_block_nums, training)
        u1 = tf.concat([u1, d1], axis=-1)
        #print(u1)

        return dense_block(u1, 12, 3, training)
        
def densenet_full(x, name='densenet_full', training=True, reuse=False):
    
    # default params
    growth_rate = 7
    
    with tf.variable_scope(name, reuse=reuse):
        # d1
        d1 = tf.layers.conv2d(x, filters=growth_rate, kernel_size=[3, 3], padding='same')
        d1 = dense_block(d1, growth_rate, 3, training)
        ##print(d1)
        
        # d2 
        d2 = down_sample(d1, growth_rate)
        d2 = dense_block(d2, growth_rate, 3, training)
        ##print(d2)

        # d3
        d3 = down_sample(d2, growth_rate)
        d3 = dense_block(d3, growth_rate, 4, training)
        ##print(d3)

        # d4
        d4 = down_sample(d3, growth_rate)
        d4 = dense_block(d4, growth_rate, 5, training)
        d4_lstm = LSTM_layer(d4, 128)
        d4 = tf.concat([d4, d4_lstm], axis=-1)
        ##print(d4)
        
        # d5
        d5 = down_sample(d4, growth_rate)
        d5 = dense_block(d5, growth_rate, 5, training)
        ##print(d5)
        
        # u4
        u4 = up_sample(d5, growth_rate)
        u4 = dense_block(u4, growth_rate, 5, training)
        u4 = tf.concat([u4, d4], axis=-1)
        ##print(u3)
    
        # u3
        u3 = up_sample(d4, growth_rate)
        u3 = dense_block(u3, growth_rate, 4, training)
        u3 = tf.concat([u3, d3], axis=-1)
        ##print(u3)

        # u2
        u2 = up_sample(u3, growth_rate)
        u2 = dense_block(u2, growth_rate, 3, training)
        u2 = tf.concat([u2, d2], axis=-1)
        u2_lstm = LSTM_layer(u2, 128)
        u2 = tf.concat([u2, u2_lstm], axis=-1)
        #print(u3)

        # u1
        u1 = up_sample(u2, growth_rate)
        u1 = dense_block(u1, growth_rate, 3, training)
        u1 = tf.concat([u1, d1], axis=-1)
        #print(u1)

        return dense_block(u1, 12, 3, training)
    
def forward(audios_mag, bands, training, reuse=False):
    # divide bands
    audios_band1 = audios_mag[:, :, bands[0]:bands[1]]
    audios_band2 = audios_mag[:, :, bands[1]:bands[2]]
    audios_band3 = audios_mag[:, :, bands[2]:bands[3]]
    audios_full = audios_mag[:, :, bands[0]:bands[3]] 

    # densenet outputs
    outputs_band1 = densenet_band1(audios_band1)
    outputs_band2 = densenet_band2(audios_band2)
    outputs_band3 = densenet_band3(audios_band3)
    outputs_full = densenet_full(audios_full)

    # concat outputs along frequency axis
    outputs = tf.concat([outputs_band1, outputs_band2, outputs_band3], axis=2)
    # concat outputs along channel axis
    outputs = tf.concat([outputs, outputs_full], axis=3)

    # last conv to adjust channels
    outputs = tf.layers.conv2d(outputs, filters=2, kernel_size=[1, 2], padding='same')
    outputs = tf.concat([outputs, audios_mag[:, :, -1:]], axis=2)
    
    return outputs

                