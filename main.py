import base64
import json
import os
import re
from datetime import datetime, timezone

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

load_dotenv()

geminiApiKey = os.getenv("GEMINI_API_KEY")
geminiModel = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

springApiUrl = os.getenv(
    "SPRING_API_URL",
    "https://lutaco-api.onrender.com"
)

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", 8000))

maxFileSizeMb = int(os.getenv("MAX_FILE_SIZE_MB", 10))
maxFileSize = maxFileSizeMb * 1024 * 1024

allowedOrigins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:4200"
).split(",")

if not geminiApiKey:
    raise ValueError("Missing GEMINI_API_KEY")

genai.configure(api_key=geminiApiKey)

app = FastAPI(
    title="Lutaco AI Bill Extractor",
    version="1.0.0",
    description="AI OCR bill extraction service"
)

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowedOrigins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

allowedTypes = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


class BillItem(BaseModel):
    """Represents a single bill item."""

    name: str
    quantity: float | None = None
    unitPrice: float | None = None
    totalPrice: float | None = None


class BillData(BaseModel):
    """Represents extracted bill information."""

    storeName: str | None = None
    storeAddress: str | None = None
    date: str | None = None
    time: str | None = None
    items: list[BillItem] = Field(default_factory=list)
    subtotal: float | None = None
    discount: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    paymentMethod: str | None = None
    category: str | None = None
    notes: str | None = None


class TransactionDraft(BaseModel):
    """Represents transaction draft data."""

    amount: int | None = None
    transactionDate: str | None = None
    note: str | None = None
    suggestedCategory: str | None = None


class ExtractResponse(BaseModel):
    """Represents extract API response."""

    success: bool
    transactionDraft: TransactionDraft | None = None
    bill: BillData | None = None
    rawText: str | None = None
    error: str | None = None


prompt = """
Bạn là AI extract thông tin từ ảnh hóa đơn / bill.

Trả về JSON THUẦN. KHÔNG markdown. KHÔNG ```json.

Schema:
{
  "storeName": string | null,
  "storeAddress": string | null,
  "date": string | null,
  "time": string | null,
  "items": [
    {
      "name": string,
      "quantity": number|null,
      "unitPrice": number|null,
      "totalPrice": number|null
    }
  ],
  "subtotal": number | null,
  "discount": number | null,
  "tax": number | null,
  "total": number | null,
  "currency": string | null,
  "paymentMethod": string | null,
  "category": string | null,
  "notes": string | null
}

Category:
food_drink | rent | utilities | transport | shopping |
healthcare | education | entertainment | other

Rules:
- amount là số nguyên
- không chứa dấu phẩy
- không chứa ký hiệu tiền tệ
- nếu không chắc thì để null
"""


def validateTokenWithSpring(authHeader: str):
    """Validate JWT token with Spring backend."""

    try:
        response = requests.get(
            f"{springApiUrl}/api/v1/users/me",
            headers={"Authorization": authHeader},
            timeout=10
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )

        body = response.json()

        if not body.get("success"):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )

        return body["data"]

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Auth service timeout"
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Auth service unavailable"
        )


def extractJson(text: str) -> dict:
    """Extract JSON from Gemini response."""

    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def buildTransactionDraft(
    bill: BillData
) -> TransactionDraft:
    """Build transaction draft from bill data."""

    amount = int(bill.total) if bill.total else None

    noteParts = []

    if bill.storeName:
        noteParts.append(bill.storeName)

    if bill.notes:
        noteParts.append(bill.notes)

    note = " - ".join(noteParts)

    return TransactionDraft(
        amount=amount,
        transactionDate=datetime.now(
            timezone.utc
        ).isoformat(),
        note=note if note else None,
        suggestedCategory=bill.category
    )


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "ok",
        "model": geminiModel,
        "maxFileSizeMb": maxFileSizeMb
    }


@app.post(
    "/extract",
    response_model=ExtractResponse
)
async def extractBill(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Extract bill information from uploaded image."""

    authHeader = f"Bearer {credentials.credentials}"

    user = validateTokenWithSpring(authHeader)

    print("USER:", user)

    if file.content_type not in allowedTypes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}"
        )

    content = await file.read()

    if len(content) > maxFileSize:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {maxFileSizeMb}MB."
        )

    try:
        model = genai.GenerativeModel(geminiModel)

        imagePart = {
            "mime_type": file.content_type,
            "data": base64.b64encode(content).decode("utf-8")
        }

        response = model.generate_content([
            prompt,
            imagePart
        ])

        rawText = response.text

        parsed = extractJson(rawText)

        bill = BillData(**parsed)

        draft = buildTransactionDraft(bill)

        return ExtractResponse(
            success=True,
            transactionDraft=draft,
            bill=bill,
            rawText=rawText
        )

    except json.JSONDecodeError:
        return ExtractResponse(
            success=False,
            rawText=rawText if "rawText" in locals() else None,
            error="Gemini returned invalid JSON"
        )

    except Exception as e:
        return ExtractResponse(
            success=False,
            rawText=rawText if "rawText" in locals() else None,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )