import cv2
import os
import glob
from PIL import Image
from skimage import exposure
import numpy as np


def tratar_imagens(pasta_origem, pasta_destino='ajeitado'):
    arquivos = glob.glob(f"{pasta_origem}/*")
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)

        resize_img_roi = cv2.resize(imagem, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Converte para escala de cinza
        img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

        # Binariza imagem
        _, img_binary = cv2.threshold(img_cinza, 105, 255, cv2.THRESH_BINARY)

        # Desfoque na Imagem
        img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)
        nome_arquivo = os.path.basename(arquivo)
        # Grava o pre-processamento para o OCR
        cv2.imwrite(f'{pasta_destino}/{nome_arquivo}', img_desfoque)

if __name__ == "__main__":
    tratar_imagens('bdcaptcha')

