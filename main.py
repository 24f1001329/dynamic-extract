from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from datetime import datetime
from openai import OpenAI
import json
import os

app = FastAPI()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://aipipe.org/openai/v1"
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
   - date
   - array[string]
   - array[integer]
7. Dates must be YYYY-MM-DD.
8. Integers and floats must be JSON numbers.
9. Booleans must be true or false.
"""


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/dynamic-extract")
def dynamic_extract(req: ExtractRequest):
    try:
        prompt = f"""
TEXT:
{req.text}

SCHEMA:
{json.dumps(req.schema, indent=2)}

Extract the fields from the text.

Return ONLY JSON matching the schema exactly.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        result = json.loads(response.choices[0].message.content)

        validated = {}

        for key, field_type in req.schema.items():
            value = result.get(key)

            if value is None:
                validated[key] = None
                continue

            try:
                if field_type == "string":
                    validated[key] = str(value)

                elif field_type == "integer":
                    validated[key] = int(value)

                elif field_type == "float":
                    validated[key] = float(value)

                elif field_type == "boolean":
                    validated[key] = bool(value)

                elif field_type == "date":
                    try:
                        dt = datetime.fromisoformat(str(value))
                        validated[key] = dt.strftime("%Y-%m-%d")
                    except:
                        validated[key] = str(value)

                elif field_type == "array[string]":
                    validated[key] = [str(x) for x in value]

                elif field_type == "array[integer]":
                    validated[key] = [int(x) for x in value]

                else:
                    validated[key] = value

            except:
                validated[key] = None

        return validated

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))