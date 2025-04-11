from transformers import BertTokenizer, BertForSequenceClassification
import torch
from os.path import join, abspath, dirname, exists 

# Define o caminho absoluto para o diretório do modelo
modelo_path = abspath(join(dirname(__file__), "./modelo_sentimentos/"))

# Verifique se o arquivo pytorch_model.bin ou model.safetensors está presente no diretório
if not exists(join(modelo_path, "pytorch_model.bin")) and not exists(join(modelo_path, "model.safetensors")):
    raise FileNotFoundError(f"Arquivo 'pytorch_model.bin' ou 'model.safetensors' não encontrado no diretório {modelo_path}")

modelo = BertForSequenceClassification.from_pretrained(modelo_path)
tokenizer = BertTokenizer.from_pretrained(modelo_path)

def analise_sentimento(text: str):
    # Tokenizer o texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Fazer predição
    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
    
    # Obter a classe com maior probabilidade e retornar a string para o usuário ao invés do label
    sentimento_classe = torch.argmax(logits, dim=1).item()
    sentiments_labels={
        0: "Admiracao",  
        1: "Diversao", 
        2: "Raiva",  
        3: "Aborrecimento", 
        4: "Aprovacao", 
        5: "Cuidadoso",  
        6: "Confusao",
        7: "Curiosidade",  
        8: "Desejo", 
        9: "Desapontamento" , 
        10: "Desaprovacao", 
        11: "Nojo", 
        12: "Embaraco", 
        13: "Exitacao", 
        14: "Medo", 
        15: "Gratidao",
        16: "Pesar",
        17: "Alegria", 
        18: "Amoor", 
        19: "Nervosismo",
        20: "Otimismo",
        21: "Orgulho",
        22: "Realizacao", 
        23: "Alivio", 
        24: "Remorso",
        25: "Tristeza",
        26: "Surpresa"  
    }
    return sentiments_labels.get(sentimento_classe, "Desconhecido")
