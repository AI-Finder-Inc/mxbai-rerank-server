from runpod import serverless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"

# Load model at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ✅ Safe fix: define pad_token if missing
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

model.eval().cuda()

def handler(event):
    query = event['input']['query']
    documents = event['input']['documents']
    pairs = [f"{query} </s> {doc}" for doc in documents]

    # ✅ Use padding and truncation safely now
    inputs = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda")

    with torch.no_grad():
        scores = model(**inputs).logits.view(-1)

    reranked = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)
    return {"reranked": reranked}

# ✅ Register the handler
serverless.start({"handler": handler})
