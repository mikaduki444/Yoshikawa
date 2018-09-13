""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

TensorFlowを使用して2層の自動エンコーダを構築し、画像を
潜在空間を小さくして再構築します。

参考文献：
    LeCun、L.Bottou、Y.Bengio、およびP.Haffner。 "グラデーションベース
    学習が文書認識に適用されている」IEEEの予稿集、
    86（11）：2278-2324、1998年11月。

試験的なニューラルネットワーク分割
中間層までのエンコードを行い、最も少ないニューロンをautoencoder_receiveへ渡す。
"""

from __future__ import division, print_function, absolute_import

#__future__ : python2でpython3の機能を使えるようにするモジュール
#割り算、print、の仕様変更、相対インポートの設定

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import autoencoder_receive as receive
#tensorflow,numpy,matplotlibのインポート
#matplotlib : グラフ描画ライブラリ

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#数字の学習データのインポート

# Training Parameters
learning_rate = 0.01
#num_steps = 30000 #学習回数
batch_size = 256  #データサイズ

display_step = 1000
examples_to_show = 10

# Network Parameters

#num_hidden_1 = 256 # 1st layer num features
num_hidden_1 = 50 #第一層
#num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_hidden_2 = 20 #第２層
#add
num_hidden_3 = 2  #第３層
num_input = 784 # MNIST data input (img shape: 28*28)
#入力データ

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
#placeholder : データを格納する。Noneを使うことで不定のまま、実行時に任意の数値を与えることができる

#重み
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2,num_hidden_3]))#,
    #'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3,num_hidden_2])),
    #'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    #'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
#傾き、偏り
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3]))#,
    #'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    #'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    #'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
# エンコーダの構築
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    # エンコーダのシグモイド関数による活性化を伴う隠しレイヤ#1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    # エンコーダのシグモイド関数による活性化を伴う隠しレイヤ#2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                               biases['encoder_b3']))
    # エンコーダのシグモイド関数による活性化を伴う隠しレイヤ#3

    #シグモイド関数:入力信号の総和から次の層に渡す値の調整（非線形）

    return layer_3
    #レイヤ3を最終層とする

#test1
"""
# Building the decoder
# デコーダの構築
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    # デコーダのシグモイド活性化を伴う隠しレイヤ#1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # デコーダのシグモイド活性化を伴う隠しレイヤ#2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    # デコーダのシグモイド活性化を伴う隠しレイヤ#3

    return layer_3
    #レイヤ3を最終層とする
"""

# Construct model
# モデルの構築
encoder_op = encoder(X)
#test1
#decoder_op = receive.decoder(encoder_op)
receive.decoder(encoder_op)
"""
# Prediction
# 予測
y_pred = decoder_op
# Targets (Labels) are the input data.
# ターゲット（ラベル）は入力データです。
y_true = X

# Define loss and optimizer, minimize the squared error
# 損失と最適化を定義し、二乗誤差を最小化する
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
# 変数を初期化する（すなわちそれらのデフォルト値を割り当てる）
init = tf.global_variables_initializer()
"""

#test2
"""
学習部分はautoencoder_receiveで行う
# Start Training
# 学習開始
# Start a new TF session
# 新しいTensorFlowのセッションの開始
with tf.Session() as sess:

    # Run the initializer
    # 変数の初期化
    sess.run(init)

    # Training
    # 学習
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    #学習
    #テストセットからの画像を符号化および復号化し、再構成を視覚化する。
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])


    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
"""
