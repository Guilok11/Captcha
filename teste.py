import os
import glob
import shutil
import easyocr

# Função para extrair a letra do nome do arquivo
def extrair_letra(nome_arquivo):
    return nome_arquivo[0].upper()

# Função para copiar a imagem para a pasta correspondente à letra
def processar_imagem(imagem_path, reader):
    nome_arquivo = os.path.basename(imagem_path)
    letras = reader.readtext(imagem_path, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if not letras:
        print(f"Não foi possível identificar uma letra na imagem {nome_arquivo}.")
        return

    letra = letras[0].upper()

    # Copiar a imagem para a pasta correspondente à letra
    pasta_destino = f"base_letras/{letra}"
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    destino = os.path.join(pasta_destino, nome_arquivo)
    shutil.copy(imagem_path, destino)
    print(f"Imagem {nome_arquivo} copiada para a pasta {pasta_destino}.")

# Caminho para a pasta com as imagens de letras já recortadas
pasta_imagens = "letras"

# Criar o objeto EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Obter lista de arquivos de imagem na pasta
arquivos_imagens = glob.glob(os.path.join(pasta_imagens, "*.png"))

if not arquivos_imagens:
    print("Nenhuma imagem encontrada na pasta.")
else:
    print(f"Encontradas {len(arquivos_imagens)} imagens na pasta.")
    # Processar cada imagem
    for imagem_path in arquivos_imagens:
        processar_imagem(imagem_path, reader)
    print("Processamento concluído.")

