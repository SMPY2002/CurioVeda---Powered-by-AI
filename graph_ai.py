import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_GRAPH")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


HEADERS = {
    "Content-Type": "application/json"
}

PROMPT_TEMPLATE = """
You are an AI that extracts structured data for charting.
Given the following text, extract numerical or tabular data and return:
- a chart title (string)
- a list of column headers (e.g., ["Year", "Population"])
- a list of rows (each row as a list of values, e.g., [["2010", 1234], ["2011", 1300]])
- a recommended chart type (bar, line, pie, histogram, scatter3d)

Respond only in this JSON format:
{
  "title": "Chart title here",
  "columns": ["Column1", "Column2"],
  "rows": [[...], [...]],
  "recommended_chart": "bar"  // or line, pie, etc.
}

Text:
"""

def query_llm_for_chart_data(text):
    prompt = PROMPT_TEMPLATE + text.strip()
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(
            GEMINI_ENDPOINT,
            headers=HEADERS,
            data=json.dumps(body)
        )

        response.raise_for_status()
        result = response.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

        # Try to extract JSON content
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        json_part = raw_text[start:end]
        parsed = json.loads(json_part)

        df = pd.DataFrame(parsed["rows"], columns=parsed["columns"])
        return {
            "df": df,
            "chart_type": parsed["recommended_chart"].lower(),
            "title": parsed["title"]
        }
    except Exception as e:
        print("[ERROR] Failed to extract chart data:", e)
        return None



def query_llm_for_chart_summary(df: pd.DataFrame, chart_type: str):
    prompt = f"""Here's a dataset:\n{df.to_csv(index=False)}\n
What is the main insight or trend in one sentence for a {chart_type} chart?"""

    body = {
        "contents": [{ "parts": [{ "text": prompt }] }]
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=HEADERS, data=json.dumps(body))
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return None
