import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard  # Importe o TensorBoard callback
from helpers import resize_to_fit

# Função para exibir informações do dataset
def print_dataset_info(name, dataset):
    print(f" - Dataset: {name}, Tipo de Dados: {dataset.dtype}")

dados = []
rotulos = []
pasta_base_imagens = "base_letras"

imagens = paths.list_images(pasta_base_imagens)

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # padronizar a imagem em 20x20
    imagem = resize_to_fit(imagem, 20, 20)

    # adicionar uma dimensão para o Keras poder ler a imagem
    imagem = np.expand_dims(imagem, axis=2)

    # adicionar às listas de dados e rótulos
    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype="float") / 255
rotulos = np.array(rotulos)

# separação em dados de treino (75%) e dados de teste (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# Converter com one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinarizer em um arquivo com o pickle dentro da pasta logs
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

with open(os.path.join(logs_dir, 'rotulos_modelo.dat'), 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criar e treinar a inteligência artificial
modelo = Sequential()

# criar as camadas da rede neural
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# criar a 2ª camada
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# mais uma camada
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))
# camada de saída
modelo.add(Dense(26, activation="softmax"))
# compilar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callback para o TensorBoard
log_dir = os.path.join(logs_dir, "logs")  # Diretório onde os logs serão salvos
tensorboard_callback = TensorBoard(log_dir=log_dir)

# Treinar o modelo e usar o callback do TensorBoard
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=10, verbose=1, callbacks=[tensorboard_callback])

# salvar o modelo em um arquivo na pasta logs
modelo.save(os.path.join(logs_dir, "modelo_treinado.hdf5"))

