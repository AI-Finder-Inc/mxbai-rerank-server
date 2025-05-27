from runpod import serverless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"

# Load model and tokenizer at container startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval().cuda()

def handler(event):
    query = event["input"]["query"]
    documents = event["input"]["documents"]

    with torch.no_grad():
        pairs = [f"{query} </s> {doc}" for doc in documents]
        inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        scores = model(**inputs).logits.view(-1)
        sorted_docs = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)

    return {"reranked": sorted_docs}

# Register the handler for RunPod Serverless
serverless.start({"handler"
