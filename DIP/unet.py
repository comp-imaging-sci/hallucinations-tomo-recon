"""
Copyright (c) Computational Imaging Science Laboratory @ UIUC
Author      : Varun A. Kelkar
Email       : vak2@illinois.edu
"""

import tensorflow as tf


def add_arguments(parser):
    parser.add_argument("--in_chans", type=int, default=1, help="Channels to the input of unet")
    parser.add_argument("--out_chans", type=int, default=1, help="Channels to the op of unet")
    parser.add_argument("--chans", type=int, default=32, help="Number of channels in each conv layer")
    parser.add_argument("--num_pool_layers", type=int, default=4, help="Number of pool layers in unet")
    parser.add_argument("--drop_prob", type=float, default=0., help="Dropout probability")
    return parser


class ConvBlock(tf.keras.Model):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.selflayers = [
            tf.keras.layers.Conv2D(out_chans, kernel_size=3, padding='same', use_bias=False, kernel_initializer='random_normal'),
            lambda x: tf.contrib.layers.instance_norm(x),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(drop_prob),
            tf.keras.layers.Conv2D(out_chans, kernel_size=3, padding='same', use_bias=False, kernel_initializer='random_normal'),
            lambda x: tf.contrib.layers.instance_norm(x),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(drop_prob),
        ]

    def call(self, inputs):

        x = inputs
        for l in self.selflayers:
            x = l(x)

        return x 

    # def __repr__(self):
    #     return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
    #         f'drop_prob={self.drop_prob})'


class TransposeConvBlock(tf.keras.Model):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.selflayers = [
            tf.keras.layers.Conv2DTranspose(out_chans, kernel_size=2, strides=(2,2), use_bias=False, kernel_initializer='random_normal'),
            lambda x: tf.contrib.layers.instance_norm(x),
            tf.keras.layers.LeakyReLU(0.2),
        ]


    def call(self, input):
        x = input
        for l in self.selflayers:
            x = l(x)

        return x

    # def __repr__(self):
    #     return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class Net(tf.keras.Model):
    """
    Tensorflow implementation of a U-Net model.
    This is based on:
        1. FastMRI UNet implementation https://github.com/facebookresearch/fastMRI/blob/master/models/unet/train_unet.py
        2. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, args):
        super().__init__()

        self.in_chans = in_chans = args.in_chans
        self.out_chans = out_chans = args.out_chans
        self.chans = chans = args.chans
        self.num_pool_layers = num_pool_layers = args.num_pool_layers
        self.drop_prob = drop_prob = args.drop_prob

        self.down_sample_layers = [ConvBlock(in_chans, chans, drop_prob)]
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = []
        self.up_transpose_conv = []
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
                ConvBlock(ch * 2, ch, drop_prob),
            ]
        self.final_conv = tf.keras.layers.Conv2D(self.out_chans, kernel_size=1, strides=(1,1), kernel_initializer='random_normal')


    def call(self, input):
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            # output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = tf.compat.v1.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(output)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            # padding = [0, 0, 0, 0]
            # if output.shape[-1] != downsample_layer.shape[-1]:
            #     padding[1] = 1 # Padding right
            # if output.shape[-2] != downsample_layer.shape[-2]:
            #     padding[3] = 1 # Padding bottom
            # if sum(padding) != 0:
            #     output = F.pad(output, padding, "reflect")

            if len(stack):
                output = tf.concat([output, downsample_layer], axis=-1)
            output = conv(output)
        
        output = self.final_conv(output)

        return output


if __name__=='__main__':

    import numpy as np 
    import time

    sess = tf.Session().__enter__()
    class args:
        pass
    args.chans = 32
    args.num_pool_layers = 4
    args.drop_prob = 0
    args.in_chans = 1
    args.out_chans = 1

    net = Net(args)
    # cblock = ConvBlock(1, 1, 0.3)
    # cblock = TransposeConvBlock(1, 1)
    x = tf.placeholder(shape=[1,256,256,1], dtype=tf.float32)
    y = net(x)
    sess.run(tf.initialize_all_variables()) 

    xnp = np.random.randn(1,256,256,1).astype(np.float32)
    ynp = sess.run(y, feed_dict={x:xnp})
    