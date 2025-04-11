from transformers import AutoModelForSequenceClassification, AutoTokenizer
from os.path import join, abspath, dirname, exists
import torch

# Define o caminho absoluto para o diretório do modelo
model_path = "./modelo_sentimentos/"
model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analise_sentimento(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

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
