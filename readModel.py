import os

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from tensorflow import keras


class modelTF:
  def readModel(img, modelo):
    class_names = ['Avião', 'automobile', 'Pássaro', 'Gato', 'Veado',
               'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']

    #importando a imagem redimensionando-a para 32x32
    image = tf.keras.utils.load_img(img,target_size=(32,32))

    #Transformando a imagem em um tensor
    input_arr = tf.keras.utils.img_to_array(image)   

    #Transformando o tensor em array numpy
    input_arr = np.array([input_arr])

    #Normalizando os dados deixando os valores entre 0 e 1
    input_arr=input_arr/255.0

    #Carrega o modelo pré treinado
    new_model = tf.keras.models.load_model(modelo)
    #Realizando a predição da classe
    predictions_single = new_model.predict(input_arr)
    

    #Escolhendo a classe com o maior valor
    for i in range(len(class_names)):
      if predictions_single[0][i]==max(predictions_single[0]):
        predict = class_names[i]
    return predict, predictions_single
