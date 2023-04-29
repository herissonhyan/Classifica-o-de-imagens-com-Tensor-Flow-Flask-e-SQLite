#  Classificação de imagens com Tensor Flow + Flask e SQLite

Essa aplicação realiza o processo de classificação de imagens com redes neurais usando a biblioteca Python [TensorFlow][df1], armazena um modelo pré-treinado no banco de dados [SQlite][df4] com [SQLAlchemy][df2] , carregar o modelo armazenado para reconhecer um novo exemplo e mostra o resultado de sua previsão em uma página HTML com o micro-framework [Flask][df3].

Grupo 1:
- Herisson Hyan
- Rangel Melo
- Luiz Carlos

As principais tecnologias utilizadas no processo foram:
- [TensorFlow][df1] (Para o treinamento da rede neural)
- [Flask][df3] (Para criar as rotas e mostrar os resultados na WEB.)
- [SQLAlchemy][df2] (Um ORM para auxiliar na manipulação do banco dados.)
- [SQLite][df4] (Implementa um mecanismo de banco de dados SQL pequeno , rápido e independente)
- [Colab][df5] (Provê a funcioanlidade de computação em nuvem)
- [Git/github][df6] (Para versionamento do código e upload no github)

# 1.Criar a rede neural
#### 1.1 Iniciando ambiente colab e instalando dependências

O primeiro passo para a criação da rede neural é iniciar um ambiente colab, devido a robustez no processamento das informações essa ferramenta se mostra muito prática, já que o google disponibiliza memória RAM e armazenamento para processamento em nuvem.
Crie um Notebook [Colab][df5] e importe as seguintes dependências.

```sh
#Para treinamento
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#para visualização
import matplotlib.pyplot as plt
```
### 1.2 Dataset
O dataset utilizado foi o CIFAR-10 esse conjunto de dados consiste em 60.000 imagens coloridas 32x32 em 10 classes, com 6.000 imagens por classe. Existem 50.000 imagens de treinamento e 10.000 imagens de teste.
Para mais informações consulte a [documentação][df7].

Para importar esse conjunto de dados no nosso notebook usamos o seguinte comando:
```sh
#Baixar dataset cifar10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar valores dos pixels entre 0 e 1
train_images, test_images = train_images / 255.0, test_images / 255.0
```
### 1.3 Rede neural convolucional.
![teste](https://matheusfacure.github.io/img/tutorial/convnet.png)

Assim como o nome sugere uma rede neural convolucional utiliza do processo de convolução para o pré-processamento das imagens.

Isso porque passar uma imagem diretamente para uma rede neural não é muito proveitoso, pelo fato da grande quantidade de informações desnecessárias, assim é preciso abstrair o que de fato é uma característica da imagem e passar essas características para a rede neural.

Para esse processo de abstração se usa a convolução2D, ela passa um pequeno filtro por toda imagem na tentar pegar características importantes. Esse processo é realizado algumas vezes aumentando as camadas que a imagem possui, após a convolução é feito o pooling, camadas de Pooling são geralmente usadas imediatamente após camadas convolucionais e o que fazem é simplificar as informações na saída da camada convolucional.

Realizado esse processo algumas vezes usa-se uma camada de flatten para achatar os dados em um único tensor, preparando os dados para o processo de aprendizado nas redes neurais.

Utilizando o TensorFlow para essse processo temos algo assim:
```sh
#Inicia uma sequencia de camadas
model = models.Sequential()
#Nota-se a não linearidade na função de ativação ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

#Os dados são tratados para entrar em uma MLP
model.add(layers.Flatten())
#512 neuronios de entrada e 10 neuronios de saidas
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10))
```
Vamos agora definir os parâmetros de treinamento:
```sh
#Aplicando o otmizador e a função de perca e após isso treinando o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels),batch_size=32)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```
treinamento feito o proxima passo é salvar esse modelo treinado para usarmos localmente em nossa aplicação através do seguinte comando:
```sh
model.save('modelo.h5')
```

# 2 Criar leitor de modelos
Agora que o modelo foi baixado é preciso escolher a imagem para fazer a predição e usar o modelo baixado:
```sh
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow import keras

#importando a imagem redimensionando-a para 32x32
image = tf.keras.utils.load_img("imagem_teste.png",target_size=(32,32))
#Transformando a imagem em um tensor
input_arr = tf.keras.utils.img_to_array(image)   
#Transformando o tensor em array numpy
input_arr = np.array([input_arr])
#Normalizando os dados deixando os valores entre 0 e 1
input_arr=input_arr/255.0
#Carrega o modelo pré treinado
new_model = tf.keras.models.load_model("modelo.h5")
#Realizando a predição da classe
predictions_single = new_model.predict(input_arr)
```
O "predictions_single" é um tensor com o valor de previsão para todas as classes, assim precisamos pegar o maior valor com o seguinte código:
```sh
#Escolhendo a classe com o maior valor
for i in range(len(class_names)):
    if predictions_single[0][i]==max(predictions_single[0]):
        predict = class_names[i]
    return predict
```
# 3 Salvando o modelo no SQLite
Agora que temos o modelo e conseguimos usá-lo em nosso projeto, precisamos salva-lo em um banco de dados, como é uma aplicação simples com apenas um valor para ser salvo optou-se utilizar SQLite que implementa um mecanismo de banco de dados SQL pequeno, rápido e independente.

## 3.1 SQLAlchemy
Como a proposta é retornar a previsão na WEB, uma boa alternativa para isso é usar o micro-framework Flask, e com isso uma biblioteca para auxiliar a manipulação do banco de dados o SQLALchemy.

## 3.2 Criação do banco de dados, tabelas e funções.

```sh
#Inicializa um app flask
app = Flask(__name__)
#Configurações do SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#definição da tabela Upload do banco de dados project.db
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

#função de criação do banco de dados e da tabela Upload.
def criartabela():
    db.create_all()
    pass

#Adição do modelo treinado na tabela Upload
def addmodel(nome,patch):
    #Primeiro deve-se converter o arquivo para binário
    def convert_into_binary(file_path):
        with open(file_path, 'rb') as file:
            binary = file.read()
        return binary
    data=convert_into_binary(patch)
    #Envio dos dados para a tabela
    upload = Upload(filename=nome, data=data)
    db.session.add(upload)
    db.session.commit()
    pass

#Criação do novo modelo.h5 a partir do banco de dados
def criarh5(file_name):
    #Query que pega o modelo da tabela 
    upload = Upload.query.filter_by(id=1).first()
    #Recriação do modelo a partir do binário do banco de dados
    with open(file_name, 'wb') as file:
        file.write(upload.data)
    pass
```
Antes de iniciar a aplicação deve-se seguir os passos para preparar o banco de dados:
 - Chamar a função de criação do banco de dados
 - ```criartabela()```
 - Adiconar o modelo pré treinado no banco de dados 
 - ```addmodel("modelo","modelo.h5")```
 - Criar modelo através do banco de dados
 - ``` criarh5("modelo_treinado/modelo.h5") ```

# 4 Aplicação Flask

Para pegar a imagem para ser feita a previsão foi criado um servidor WEB com um HTML responsavel por passar os dados para as funções de previsões.
```sh
#É criado a rota de GET e renderizado um template HTML para passar a imagem para a função de previsão
@app.route("/", methods=["GET"])
def index():
    #criartabela()
    #addmodel("modelo","modelo.h5")
    #criarh5("modelo_treinado/modelo.h5")
    return render_template('index.html')

#É criado a rota POST para receber a informação, passar para a função de previsão e retornar os resultados para o usuário
@app.route("/", methods=["POST"])
def post_file():
    arquivo=request.files.get("minhaImage")
    patch="images/"+arquivo.filename
    arquivo.save(patch)
    retorno=modelTF.readModel(patch,"modelo_treinado\modelo.h5")
    predicts=retorno[1][0]
    previsao=retorno[0]
    #class_names = ['Avião', 'automobile', 'Pássaro', 'Gato', 'Veado',
    #          'Cachorro', 'Sapo', 'Cavalo', 'Navio', 'Caminhão']
    return (f"<h1>O resultado da previsão foi: {previsao}</h1> <h2>Avião: {predicts[0]}</h2> <h2>Carro: {predicts[1]}</h2> <h2>pássaro:{predicts[2]}</h2> <h2>Gato: {predicts[3]}</h2> <h2>Veado: {predicts[4]}</h2> <h2>Cachorro: {predicts[5]}</h2> <h2>Sapo: {predicts[6]}</h2> <h2>Cavalo: {predicts[7]}</h2> <h2>Navio: {predicts[8]}</h2> <h2>Caminhão: {predicts[9]}</h2>")
```
HTML para envio da imagem através da rota POST:
```sh
<form action="/" method="post" enctype="multipart/form-data">
        <h1>Escolha uma imagem</h1>
        <input type="file" name="minhaImage" id="minhaImage">
        <button class="Submit">Enviar</button>
    </form>
```

Assim criamos uma aplicação que recebe os dados atraves de um HTML, lê um modelo pré treinado de um banco de dados e retorna a previsão para o usuário.

:)


[df1]: <https://www.tensorflow.org/>
[df2]: <https://www.sqlalchemy.org/>
[df3]: <https://flask.palletsprojects.com/en/2.2.x/>
[df4]:<https://www.sqlite.org/index.html>
[df5]: <https://colab.research.google.com/>
[df6]: <https://git-scm.com/downloads>
[df7]: <https://www.tensorflow.org/datasets/catalog/cifar10>
[doc]:<https://pokeapi.co/docs/v2>
