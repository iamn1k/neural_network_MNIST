import numpy as np     # работа с математикой 
import keras		   # высокоуровненвая нейронная сеть
from keras.datasets import mnist # наборы данных
from keras.models import Sequential # набор слоев
from keras.layers import Dense, Dropout # характеристика каждого слоя
from keras.utils import to_categorical # Преобразует вектор класса (целые числа) в двоичную матрицу классов.
from PIL import Image # работа с изображениями
import os			  # работа с os системой
import matplotlib.pyplot as plt # отображение изображений на экране
from draw import draw


batch_size = 512 # кол-во единиц данных, которое подается за раз на обучение
num_classes = 10 # кол - во вариантов классификации
epochs = 6 # 10	 # количество эпох (прохождений обучения)
img_size = (28, 28) # размер изображений (28 х 28 px)
img_size_flat = np.prod(img_size) # произведение элементов массива по заданной оси.

def get_image(url):
    img_size = (28, 28) # размер изображения
    target_size=img_size 
    convert_to='L'  # конвертировать изображение в черно-белое с 8 бит памяти на каждый пиксель	
    img = (Image.open(url).convert(convert_to).resize(target_size)) # .open -> открыть изображение, . convert -> конвертировать
	# . resize -> сжать до размеров 28 x 28 px    
    return np.array(img).reshape(-1, np.prod(target_size))# вернуть изображение в качестве массива 
    # .reshape -> массив изменен в одномерный


def predict_digit(img, model):
    return model.predict(img).argmax() # .predict предсказать метку наборов данных, .argmax -> индексы макс значений по оси

if not(os.path.isfile('mnist_model.h5')): # если нейронная сеть не обучена

    (x_train, y_train), (x_test, y_test) = mnist.load_data() # загружаем тренировочные данные
    x_train = x_train.reshape(-1, img_size_flat).astype('float32') / 255 # .reshape -> аналогично
    x_test = x_test.reshape(-1, img_size_flat).astype('float32') / 255 #  . astype -> к типу вещ чисел

    y_train = to_categorical(y_train, num_classes) # тренировочные изображения в матрицу
    y_test = to_categorical(y_test, num_classes) #  с 10 вариантами ответов

    model = Sequential()
    model.add(Dense(784, activation='relu', input_shape=(784,)))	# слой из 1024 нейронов, функции активации relu, 784 px
    model.add(Dropout(0.2)) # шаг обучения
    model.add(Dense(588, activation='selu')) # tahn
    model.add(Dropout(0.2))
    model.add(Dense(294, activation='relu')) # 1024
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax')) # выходные 10 слоев, фунция активации softmax
    model.add(Dropout(0.15)) # test 
    model.compile(loss='categorical_crossentropy', # уменьшение значения потерь
              optimizer='adam', 				   # оптимизация модели с помощью метода adam
              metrics=['accuracy']) # список метрик, которые оцениваются моделью во время обучения
    # обучение модели для фиксированного числа эпох
    history = model.fit(x_train, y_train,  # данные для обучения
                    batch_size=batch_size, # кол-во данных в единицу времени
                    epochs=epochs,		   # кол-во эпох обучения
                    verbose=1,			   # один = индикатор выполнения
                    validation_data=(x_test, y_test)) # тестовые данные для проверки
    print('Learning is all')

    model.save('mnist_model.h5') # сохранение обученной модели
else: # если нейронная сеть обучена
    draw() # функция для рисования цифр, импортируется из файла draw
    model = keras.models.load_model('mnist_model.h5') # загружаем модель

    img = get_image("image.png")# получаем на вход новое нарисованное изображение
	#plt.imshow(img.reshape(img_size), cmap='gray')

    img_tensor = get_image("image.png")# преобразование к нужному для нейронной сети виду
	# >>> image tensor shape: (1, 784)
    print(predict_digit(img_tensor, model)) # вывод результата
	#input()
	
    

