from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Dropout,
    ConvLSTM2D, BatchNormalization, Activation, LeakyReLU, TimeDistributed
)

def STORM(input_shape=(4, 480, 480, 1), mode="regression", dropout=0.5):
    """
    DL STORM model.

    Args:
        input_shape (tuple): shape of input data (time series length, length, width, channels).
        mode (str): "regression" (precipitation rate) or "segmentation" (binary precipitation prediction).
        dropout (float): dropout rate.

    Returns:
        tensorflow.keras.models.Model: STORM model.
    """
    # Encoder
    inputs = Input(input_shape)
    
    conv1 = TimeDistributed(Conv2D(16, 3, padding="same", kernel_initializer="he_normal", activation="relu"))(inputs)
    conv1 = TimeDistributed(Conv2D(16, 3, padding="same", kernel_initializer="he_normal", activation="relu"))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = TimeDistributed(Conv2D(32, 3, padding="same", kernel_initializer="he_normal", activation="relu"))(pool1)
    conv2 = TimeDistributed(Conv2D(32, 3, padding="same", kernel_initializer="he_normal", activation="relu"))(conv2)
    drop2 = TimeDistributed(Dropout(dropout))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop2)

    # Bottleneck
    convLSTM3 = ConvLSTM2D(filters=64, kernel_size=(7, 7), padding='same', activation="relu", return_sequences=False)(pool2)
    bn3 = BatchNormalization()(convLSTM3)
    drop3 = Dropout(dropout)(bn3)

    # Decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(drop3), conv2[:, -1, :, :, :]], axis=3)
    conv4 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal", activation="relu")(up4)
    conv4 = Conv2D(32, 3, padding="same", kernel_initializer="he_normal", activation="relu")(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1[:, -1, :, :, :]], axis=3)
    conv5 = Conv2D(16, 3, padding="same", kernel_initializer="he_normal", activation="relu")(up5)
    conv5 = Conv2D(16, 3, padding="same", kernel_initializer="he_normal", activation="relu")(conv5)
    
    if mode == "regression":
        outputs = Conv2D(1, 1, activation='linear')(conv5)
    elif mode == "segmentation":
        outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model
    