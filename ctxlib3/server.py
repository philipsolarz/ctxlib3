from fastapi import FastAPI, HTTPException, Depends
from docarray.documents import TextDoc
from docarray.base_doc import DocArrayResponse
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
api = FastAPI()

encoder = Llama(
    model_path="/data/llama-2-7b-chat.Q4_K_M.gguf",
    verbose=True,
    n_gpu_layers=30,
    n_threads= 16,
    #use_mlock= True,
    use_mmap= True,
    n_ctx=2000,
    embedding= True
)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

class InputModel(TextDoc):
    pass

class OutputModel(TextDoc):
    pass

@api.post("/embeddings/generate/", response_model=list[OutputModel], response_class=DocArrayResponse)
def generate_embeddings(ctx: InputModel) -> list[OutputModel]:
    # Split the text into chunks.
    chunks = text_splitter.split(ctx.text)
    # Encode the chunks.
    embeddings = encoder.encode(chunks)
    # Return the embeddings.
    return [OutputModel(text=chunk, embedding=embedding) for chunk, embedding in zip(chunks, embeddings)]
