from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"

# Load model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval().cuda()

app = FastAPI()

class RerankRequest(BaseModel):
    query: str
    documents: list[str]

@app.post("/rerank")
async def rerank(req: RerankRequest):
    with torch.no_grad():
        pairs = [f"{req.query} </s> {doc}" for doc in req.documents]
        inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        scores = model(**inputs).logits.view(-1)
        sorted_docs = sorted(zip(req.documents, scores.tolist()), key=lambda x: x[1], reverse=True)
    return {"reranked": sorted_docs}
