from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense
from keras.utils import up_utils
import keras
import numpy as np
from tensorflow.python import tf2

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def main():
    X_train, X_test, y_train, y_test = np.load("./animal.npy", allow_pickle=True)
    # 正規化
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    # one-hot-vector
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_val(model, X_test, y_test)

def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) # 一次元化
    model.add(Dense(512)) #全結合層
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes)) # 最後の出力はクラスの分類の数で行う
    model.add(Activation('softmax'))

    # optimizer
    opt = tf2.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, batch_size=32, nb_epoch=50)

    # モデルと重みの保存
    model.save('./animal_cnn.h5')

    return model

def model_val(model, X, y):
    scores = model.evaluete(X, y, verbose=1)
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])

if __name__ == '__main__': # コマンドラインから実行されたら，の意味（__name_には別ファイルから読み込まれるとそのファイル名が，コマンドラインから読み込まれると__main__が格納される）
    main()
