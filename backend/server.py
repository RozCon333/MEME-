from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import pandas as pd
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import cv2
import numpy as np
from collections import Counter
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class MemeOCR(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    extracted_text: str
    keywords: List[str]  # New: extracted keywords
    word_count: int
    image_data: str  # base64 encoded
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GeneratedMeme(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    image_data: str  # base64 encoded with overlay
    source_words: List[str]
    keyword_pattern: str  # New: pattern used for generation
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GenerateMemeRequest(BaseModel):
    count: int = 5


# Helper Functions for Image Processing and Keyword Extraction

def preprocess_image_for_ocr(image):
    """Enhance low-res images for better OCR"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Upscale if too small (helps with low-res)
    height, width = gray.shape
    if width < 500 or height < 500:
        scale = max(500 / width, 500 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to PIL
    return Image.fromarray(sharpened)


def extract_keywords(text, min_length=3, max_keywords=10):
    """Extract meaningful keywords from text - KEEPS NSFW WORDS"""
    # Common words to ignore (REMOVED NSFW words - we WANT to keep those!)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'that', 'this',
        'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'them', 'their'
    }
    
    # Clean and tokenize (keep NSFW words!)
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    # Filter words - KEEP explicit content
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Get most frequent keywords
    word_counts = Counter(keywords)
    top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return top_keywords


def build_keyword_patterns(all_keywords):
    """Build patterns from keyword combinations"""
    patterns = []
    
    # Single keywords
    patterns.extend(all_keywords[:20])
    
    # Keyword pairs (common patterns)
    keyword_pairs = []
    for i, kw1 in enumerate(all_keywords[:15]):
        for kw2 in all_keywords[i+1:16]:
            keyword_pairs.append(f"{kw1}+{kw2}")
    
    patterns.extend(keyword_pairs[:10])
    
    return patterns


# Routes
@api_router.get("/")
async def root():
    return {"message": "Meme OCR & Generator API"}


@api_router.post("/upload-memes")
async def upload_memes(files: List[UploadFile] = File(...)):
    """Upload multiple meme images and perform OCR"""
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Perform OCR
            extracted_text = pytesseract.image_to_string(image)
            word_count = len(extracted_text.split())
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create meme OCR object
            meme_ocr = MemeOCR(
                filename=file.filename,
                extracted_text=extracted_text.strip(),
                word_count=word_count,
                image_data=img_base64
            )
            
            # Save to database
            doc = meme_ocr.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.meme_ocr.insert_one(doc)
            
            results.append({
                "filename": file.filename,
                "extracted_text": extracted_text.strip(),
                "word_count": word_count,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {"uploaded": len(files), "results": results}


@api_router.get("/ocr-results")
async def get_ocr_results():
    """Get all OCR results"""
    results = await db.meme_ocr.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO timestamps back
    for result in results:
        if isinstance(result['timestamp'], str):
            result['timestamp'] = datetime.fromisoformat(result['timestamp'])
    
    return results


@api_router.get("/download-csv")
async def download_csv():
    """Download OCR results as CSV"""
    results = await db.meme_ocr.find({}, {"_id": 0, "image_data": 0}).to_list(1000)
    
    if not results:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    csv_data = df.to_csv(index=False)
    
    return {"csv": csv_data}


@api_router.post("/generate-new-memes")
async def generate_new_memes(request: GenerateMemeRequest):
    """Generate new adult humor memes using LLM"""
    
    # Get all extracted texts
    ocr_results = await db.meme_ocr.find({}, {"_id": 0}).to_list(1000)
    
    if not ocr_results:
        raise HTTPException(status_code=404, detail="No meme data available. Please upload memes first.")
    
    # Collect all words
    all_words = []
    for result in ocr_results:
        words = result['extracted_text'].split()
        all_words.extend(words)
    
    # Create unique word list
    unique_words = list(set(all_words))
    word_sample = unique_words[:100]  # Sample for LLM
    
    # Initialize LLM chat
    api_key = os.environ.get('EMERGENT_LLM_KEY')
    chat = LlmChat(
        api_key=api_key,
        session_id=str(uuid.uuid4()),
        system_message="You are a creative meme generator that creates adult humor memes. Be edgy, witty, and humorous."
    ).with_model("openai", "gpt-4o")
    
    # Generate memes
    word_list = ', '.join(word_sample[:50])
    prompt = f"""Based on these words from existing memes: {word_list}

Generate {request.count} NEW adult humor meme texts. Each meme should be:
- Short (1-3 lines max)
- Funny and edgy
- Suitable for adult humor
- Creative combinations of concepts

Format your response as a JSON array of objects with 'text' and 'source_words' (array of 3-5 relevant words used).
Example: [{{"text": "When you...", "source_words": ["word1", "word2", "word3"]}}]

Return ONLY the JSON array, no other text."""
    
    user_message = UserMessage(text=prompt)
    response = await chat.send_message(user_message)
    
    # Parse LLM response
    try:
        # Clean response
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        meme_data = json.loads(response_text)
        
        # Get random images from uploaded memes for overlay
        sample_images = await db.meme_ocr.find({}, {"_id": 0, "image_data": 1}).to_list(request.count)
        
        generated_memes = []
        for idx, meme in enumerate(meme_data):
            # Use one of the uploaded images
            if idx < len(sample_images):
                img_data = sample_images[idx]['image_data']
            else:
                img_data = sample_images[0]['image_data']
            
            # Create text overlay on image
            img_bytes = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_bytes))
            
            # For now, just use the original image (text overlay requires additional setup)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            final_img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            generated_meme = GeneratedMeme(
                text=meme['text'],
                image_data=final_img_base64,
                source_words=meme.get('source_words', [])
            )
            
            # Save to database
            doc = generated_meme.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.generated_memes.insert_one(doc)
            
            generated_memes.append({
                "id": generated_meme.id,
                "text": generated_meme.text,
                "image_data": final_img_base64,
                "source_words": generated_meme.source_words
            })
        
        return {"generated": len(generated_memes), "memes": generated_memes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memes: {str(e)}")


@api_router.get("/generated-memes")
async def get_generated_memes():
    """Get all generated memes"""
    memes = await db.generated_memes.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO timestamps back
    for meme in memes:
        if isinstance(meme['timestamp'], str):
            meme['timestamp'] = datetime.fromisoformat(meme['timestamp'])
    
    return memes


@api_router.delete("/clear-data")
async def clear_data():
    """Clear all data"""
    await db.meme_ocr.delete_many({})
    await db.generated_memes.delete_many({})
    return {"message": "All data cleared"}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()