import cv2
import os
import glob

arquivos = glob.glob('ajeitado/*')
for arquivo in arquivos:
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    # em preto e branco
    _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)

    # encontrar os contornos de cada letra
    contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regiao_letras = []

    # filtrar os contornos que são realmente de letras
    for contorno in contornos:
        (x, y, largura, altura) = cv2.boundingRect(contorno)
        area = cv2.contourArea(contorno)
        if area > 115:
            regiao_letras.append((x, y, largura, altura))

    if len(regiao_letras) != 5:
        continue

    # desenhar os contornos e separar as letras em arquivos individuais

    imagem_final = cv2.merge([imagem] * 3)

    for i, retangulo in enumerate(regiao_letras):
        x, y, largura, altura = retangulo

        # Certifique-se de que as coordenadas e dimensões do retângulo sejam positivas
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if largura < 0:
            largura = 0
        if altura < 0:
            altura = 0

        # Recorte a imagem da letra usando as coordenadas e dimensões corretas
        imagem_letra = imagem[y:y+altura, x:x+largura]

        nome_arquivo = os.path.basename(arquivo).replace(".png", f"letra{i+1}.png")
        cv2.imwrite(f'letras/{nome_arquivo}', imagem_letra)

        cv2.rectangle(imagem_final, (x, y), (x+largura, y+altura), (0, 255, 0), 1)

    nome_arquivo = os.path.basename(arquivo)
    cv2.imwrite(f"identificado/{nome_arquivo}", imagem_final)

