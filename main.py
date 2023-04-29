import numpy as np
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

from readModel import modelTF

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
db = SQLAlchemy(app)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

def criartabela():
    db.create_all()
    pass

def addmodel(nome,patch):
    def convert_into_binary(file_path):
        with open(file_path, 'rb') as file:
            binary = file.read()
        return binary
    data=convert_into_binary(patch)
    upload = Upload(filename=nome, data=data)
    db.session.add(upload)
    db.session.commit()
    pass

def criarh5(file_name):
    upload = Upload.query.filter_by(id=1).first()
    with open(file_name, 'wb') as file:
        file.write(upload.data)
    pass

@app.route("/", methods=["GET"])
def index():
    #criartabela()
    #addmodel("modelo","modelo.h5")
    #criarh5("modelo_treinado/modelo.h5")
    return render_template('index.html')

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
    return (f"<h1>O resultado da previsão foi: {previsao}</h1> <h2>Avião: {predicts[0]}</h2> <h2>Carro: {predicts[1]}</h2> <h2>pássaro: {predicts[2]}</h2> <h2>Gato: {predicts[3]}</h2> <h2>Veado: {predicts[4]}</h2> <h2>Cachorro: {predicts[5]}</h2> <h2>Sapo: {predicts[6]}</h2> <h2>Cavalo: {predicts[7]}</h2> <h2>Navio: {predicts[8]}</h2> <h2>Caminhão: {predicts[9]}</h2>")



if __name__ == '__main__':
    app.run(port=3000)