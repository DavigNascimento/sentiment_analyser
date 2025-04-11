# API de Análise de Sentimentos - V1

Está é uma API desenvolvida com FastAPI para realizar análise de sentimentos em textos. A API recebe um texto como entrada e retornar o sentimento associado.

## Requisitos 

- Python 3.11 
- FastAPI 
- Uvicorn 
- Outros pacotes listados no arquivo `requirements.txt`

## Instalação 

1. Clone este repostiório
```
git clone https://github.com/DaviGNascimento/sentiment_analyser.git
cd api_sentiment_analyse
```

2. Crie um ambiente virtual 
```
python -m venv env
```

3. Ativie o ambiente virtual 
```
.\env\Scripts\activate
```

5. Instale as dependências 
```
pip install -r requirements.txt
```
6. Baixe o modelo.
- Baixe a [pasta no Drive](https://drive.google.com/drive/folders/1U34vGuRdZ8TiBAKsq1PsN2BLRoQExIB_?usp=drive_link) e mova-a para o diretório raiz sob o nome de "modelo_sentimentos/".

7. Inicie o servidor
```
uvicorn app.main:app --reload
```

8. Acesse a documentação interativa da API:

- Abra o navegador e vá para: http://127.0.0.1:8000/docs

## Como usar

### Endpoint `/analisar`

- Método: `POST`
- URL: `http://127.0.0.1:8000/analisar` 
- Corpo da Requisição: Envie um JSON com o seguinte formato:
```
{
    "text": "Estou muito feliz com o serviço."
}
```

- Resposta: A API retornará um JSON com o sentimento identificado:
```
{
    "sentiment": "Satisfeito"
}
```

### Exemplos de Teste

#### Usando `curl`

```
curl -X POST "http://127.0.0.1:8000/analisar" -H "Content-Type: application/json" -d "{\"text\":\"Estou muito feliz com o serviço.\"}"
```

#### Usando o Postman

1. Abra o Postman
2. Crie uma nova requisição
3. Selecione o método `POST`
4. No campo de URL, insira:  `http://127.0.0.1:8000/analisar`.
5. Vá até a aba `Body`.
6. Selecione a opção `raw` e escolha o tipo `JSON`.
7. Insira o seguinte JSON no corpo da requisição:

```
{
    "text": "Estou muito feliz com o serviço."
}
```

8. Clique em `Send`.

### Estrutura do projeto

```
api_sentiment_analyse/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── modelo_sentimentos/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── outros arquivos do modelo...
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```
