### RAG (Retrieval Augmented Generation) Demo



### Follow these steps to run the RAG demo on your local machine:

1. Create an example.env to store your API huggingface token as:
```This is a token that you can get from your huggingface account. You need to create an account and get the token from the settings.```

```bash
HUGGINGFACEHUB_API_TOKEN= ""
```

2. Create data/ directory to store the pdf files that you want to use for the RAG demo.


3. Create a virtual environment and install the requirements:
```bash
pip install -r requirements.txt
```

4. Download LLM model of your choice locally from the hugging face. For this project following models are recommended as:

- gemma-2b-it.Q2_K.gguf
- Phi-3-mini-4k-instruct-q4.gguf
- mistral-7b-v0.1.Q2_K.gguf
