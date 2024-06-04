### RAG (Retrieval Augmented Generation) Demo



### Follow these steps to run the RAG demo on your local machine:

👉 Create an example.env to store your API huggingface token as:

_Note: This is a token that you can get from your huggingface account. You need to create an account and get the token from the settings._

```bash
HUGGINGFACEHUB_API_TOKEN= ""
```

👉 Create data/ directory to store the pdf files that you want to use for the RAG demo.


👉 Create a virtual environment and install the requirements:
```bash
pip install -r requirements.txt
```

👉 Download LLM model of your choice locally from the hugging face. For this project following models are recommended as:

- [gemma-2b-it.Q2_K.gguf](https://huggingface.co/asedmammad/gemma-2b-it-GGUF/tree/main)
- [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main)
- [mistral-7b-v0.1.Q2_K.gguf](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/tree/main)


👉 Download llama.cpp and compile it using the following command for the GPU support:

For NVIDIA GPU:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

For Apple Metal M1/M2 GPU support:
```bash
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

