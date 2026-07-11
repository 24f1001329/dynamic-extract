from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import json
import os

from openai import OpenAI

app = FastAPI()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

class ExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]


SYSTEM_PROMPT = """
You are a data extraction engine.

Rules:
1. Return ONLY valid JSON.
2. Return EXACTLY the keys provided in the schema.
3. No extra keys.
4. No missing keys.
5. If a value cannot be found, return null.
6. Supported types:
   - string
   - integer
   - float
   - boolean
   - date (YYYY-MM-DD)
   - array[string]
   - array[integer]
7. Integers and floats must be JSON numbers.
8. Booleans must be true/false.
9. Dates must be ISO format YYYY-MM-DD.
"""


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/dynamic-extract")
def dynamic_extract(req: ExtractRequest):
    try:
        prompt = f"""
TEXT:
{req.text}

SCHEMA:
{json.dumps(req.schema, indent=2)}

Return only JSON matching the schema exactly.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        validated = {}

        for key, typ in req.schema.items():
            value = result.get(key)

            if value is None:
                validated[key] = None
                continue

            if typ == "string":
                validated[key] = str(value)

            elif typ == "integer":
                validated[key] = int(value)

            elif typ == "float":
                validated[key] = float(value)

            elif typ == "boolean":
                validated[key] = bool(value)

            elif typ == "date":
                try:
                    dt = datetime.fromisoformat(str(value))
                    validated[key] = dt.strftime("%Y-%m-%d")
                except:
                    validated[key] = str(value)

            elif typ == "array[string]":
                validated[key] = [str(x) for x in value]

            elif typ == "array[integer]":
                validated[key] = [int(x) for x in value]

            else:
                validated[key] = value

        return validated

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))