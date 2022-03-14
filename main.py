import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa

# from kaggledatasets import KaggleDatasets
import matplotlib.pyplot as plt
import numpy as np

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__)

MONET_FILENAMES = ["C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/monet_tfrec/monet00-60.tfrec",
            "C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/monet_tfrec/monet04-60.tfrec",
            "C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/monet_tfrec/monet08-60.tfrec",
            "C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/monet_tfrec/monet12-60.tfrec",
            "C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/monet_tfrec/monet16-60.tfrec"]

PHOTO_FILENAMES = [
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo00-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo01-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo02-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo03-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo04-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo05-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo06-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo07-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo08-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo09-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo10-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo11-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo12-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo13-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo14-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo15-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo16-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo17-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo18-352.tfrec",
"C:/Users/fabia/Desktop/Git/FabiCoding/Monet_GAN/gan-getting-started/photo_tfrec/photo19-352.tfrec"]


IMAGE_SIZE = [256, 256]


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image


def load_dataset(filenames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset


monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(1)
photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)

example_monet = next(iter(monet_ds))
example_photo = next(iter(photo_ds))

plt.subplot(121)
plt.title('Photo')
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet')
plt.imshow(example_monet[0] * 0.5 + 0.5)

plt.show()

OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_instamcenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instamcenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def Generator():
    inputs = layers.Input(shape=[256, 256, 3])

    #bs = batch size
    down_stack = [
        downsample(64, 4, apply_instamcenorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


with strategy.scope():
    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    photo_generator = Generator()  # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator()  # differentiates real photos and generated photos


to_monet = monet_generator(example_photo)

plt.subplot(1, 2, 1)
plt.title("Original Photo")
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(1, 2, 2)
plt.title("Monet-esque Photo")
plt.imshow(to_monet[0] * 0.5 + 0.5)
plt.show()


