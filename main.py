from fastapi import FastAPI
from llm_inference.translation.seed_x import translate

from data import TranslateReq

app = FastAPI()


async def _translate(req: TranslateReq):
    return translate(
        sentence=req.sentence,
        target_lang=req.target_lang,
        resample=req.resample,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        seed=req.seed,
        max_tokens=req.max_tokens,
        min_tokens=req.min_tokens
    )
