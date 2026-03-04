import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services import chart_generator

app = FastAPI(title="Data Visual")

#This is how backend communicates with the frontend. 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/data-analysis")
async def analyze(file: UploadFile = File(...)):
    return await chart_generator.analyze(file)

