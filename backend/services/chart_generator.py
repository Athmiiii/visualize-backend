import io
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fastapi import UploadFile, File
from openai import OpenAI
from dotenv import load_dotenv

ypoints = np.array([3, 8, 1, 10])

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"))

async def analyze(file: UploadFile = File(...)):
    content_edu= await file.read()
    df_edu= pd.read_csv(io.BytesIO(content_edu))
    sample_edu= df_edu.head(10)
    sample_edu_string= str(sample_edu)
    system_prompt = """You are a data visualization expert.
      Analyse the schema and the included sample data and suggest the ONE most useful visualization that would provide meaningful
      insight. Respond ONLY with a valid JSON format as follows:
    {
    "suggestion": [
        {
            "chart_type": "bar,line or scatter",
            "title": "...",
            "x_column": "...",
            "y_column": "..."
        }
    ]
    }

    Only suggest a chart type if the data supports it.
    """
    responsee = client.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"This is the data:\n\n{sample_edu_string}"}
    ],temperature=0.45
    )
    response= responsee.choices[0].message.content
    response_string= str(response)
    json_object = json.loads(response_string)
    graph_suggest= json_object["suggestion"][0]
    chart_type= graph_suggest["chart_type"].lower()
    title= graph_suggest["title"]
    x_col= graph_suggest["x_column"]
    y_col= graph_suggest["y_column"]
    plt.figure(figsize=(10,5))
    if "bar" in chart_type:
        sns.barplot(data=df_edu,x=graph_suggest["x_column"],y=graph_suggest["y_column"])
    elif "line" in chart_type:
        sns.lineplot(data=df_edu,x=graph_suggest["x_column"],y=graph_suggest["y_column"])
    elif "scatter" in chart_type:
        sns.scatterplot(data=df_edu,x=graph_suggest["x_column"],y=graph_suggest["y_column"])
    else:
        return("Big Error: We don't have a graph for this. More to come soon.")
    
    plt.title(graph_suggest["title"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



    
    

