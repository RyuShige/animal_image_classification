from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense
from keras.utils import up_utils
import keras
import numpy as np
import tensorflow
from PIL import Image

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)))
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

    # モデルのロード
    model = load_model('./animal_cnn.h5')

    return model

def main():
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image) #数字データに変換
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print()
