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
from spellchecker import SpellChecker

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
    corrected_text: str  # Auto-corrected version
    keywords: List[str]  # Extracted keywords
    word_count: int
    image_data: str  # base64 encoded
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ToneSettings(BaseModel):
    naughty: int = 5  # 1-10
    sexy: int = 5
    funny: int = 5
    rude: int = 5

class TonePreset(BaseModel):
    name: str
    naughty: int
    sexy: int
    funny: int
    rude: int
    description: str

class MemeStyle(BaseModel):
    length: str = "short"  # short, medium, long
    format: str = "statement"  # statement, question, observation, setup_punchline

class GeneratedMeme(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    image_data: str  # base64 encoded with overlay
    source_words: List[str]
    keyword_pattern: str
    tone_used: ToneSettings
    style_used: Optional[MemeStyle] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GenerateMemeRequest(BaseModel):
    count: int = 4  # Default 4 options
    tone: Optional[ToneSettings] = None
    style: Optional[MemeStyle] = None
    keywords: Optional[List[str]] = None  # Custom keywords

class GenerateSimilarRequest(BaseModel):
    meme_id: str
    tone: Optional[ToneSettings] = None

class UpdateTextRequest(BaseModel):
    id: str
    corrected_text: str


# Helper Functions for Image Processing and Keyword Extraction

# Initialize spell checker
spell = SpellChecker()

# Common NSFW words to always keep
NSFW_WHITELIST = {
    'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitting', 'shitty', 
    'damn', 'damned', 'sex', 'sexy', 'sexual', 'boobs', 'tits', 'titties', 
    'ass', 'asses', 'dick', 'dicks', 'cock', 'cocks', 'bitch', 'bitches',
    'pussy', 'pussies', 'cum', 'cumming', 'porn', 'horny', 'nude', 'naked',
    'bastard', 'hell', 'piss', 'crap', 'slut', 'whore', 'penis', 'vagina'
}

def auto_correct_text(text):
    """Auto-correct obvious OCR mistakes"""
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Keep short words and numbers as-is
        if len(word) <= 2 or word.isdigit():
            corrected_words.append(word)
            continue
        
        # Clean word for checking
        clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
        
        if not clean_word:
            corrected_words.append(word)
            continue
        
        # ALWAYS keep NSFW words!
        if clean_word in NSFW_WHITELIST:
            corrected_words.append(word)
            continue
        
        # Check if misspelled
        if spell.unknown([clean_word]):
            # Get correction
            correction = spell.correction(clean_word)
            
            # Only use correction if it's valid and different
            if correction and correction != clean_word and correction in spell:
                # Preserve original capitalization
                if word[0].isupper():
                    correction = correction.capitalize()
                corrected_words.append(correction)
            else:
                # If no good correction, skip the word (it's probably garbage OCR)
                continue
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

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
    """Extract meaningful keywords from text - KEEPS NSFW WORDS, FILTERS GARBAGE"""
    # Common words to ignore
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'that', 'this',
        'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'them', 'their',
        'when', 'where', 'who', 'what', 'why', 'how', 'your', 'my', 'our'
    }
    
    # Clean and tokenize (keep NSFW words!)
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    # Filter words - KEEP explicit content, REMOVE garbage OCR
    keywords = []
    for word in words:
        if len(word) < min_length or word in stop_words:
            continue
        
        # ALWAYS keep NSFW words
        if word in NSFW_WHITELIST:
            keywords.append(word)
            continue
        
        # Check if it's a real word (filter garbage OCR)
        if not spell.unknown([word]):
            keywords.append(word)
    
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


@api_router.get("/tone-presets")
async def get_tone_presets():
    """Get predefined tone presets"""
    presets = [
        {"name": "Sarcastic AF", "naughty": 6, "sexy": 3, "funny": 9, "rude": 7, "description": "Eye-rolling sarcasm"},
        {"name": "Super Horny", "naughty": 10, "sexy": 10, "funny": 6, "rude": 5, "description": "Maximum horniness"},
        {"name": "Savage Roast", "naughty": 5, "sexy": 2, "funny": 8, "rude": 10, "description": "Brutal insults"},
        {"name": "Dark Humor", "naughty": 8, "sexy": 4, "funny": 9, "rude": 8, "description": "Twisted and dark"},
        {"name": "Flirty Tease", "naughty": 7, "sexy": 9, "funny": 7, "rude": 3, "description": "Playful seduction"},
        {"name": "Angry Rant", "naughty": 6, "sexy": 2, "funny": 5, "rude": 10, "description": "Pissed off energy"},
        {"name": "Dad Jokes NSFW", "naughty": 4, "sexy": 3, "funny": 10, "rude": 4, "description": "Corny but dirty"},
        {"name": "Passive Aggressive", "naughty": 5, "sexy": 2, "funny": 7, "rude": 9, "description": "Subtle shade"},
        {"name": "Drunk Thoughts", "naughty": 8, "sexy": 7, "funny": 9, "rude": 6, "description": "3AM vibes"},
        {"name": "Wholesome Horny", "naughty": 7, "sexy": 8, "funny": 8, "rude": 2, "description": "Sweet but spicy"},
        {"name": "Existential Crisis", "naughty": 5, "sexy": 3, "funny": 8, "rude": 6, "description": "Deep thoughts"},
        {"name": "Petty Queen", "naughty": 6, "sexy": 5, "funny": 7, "rude": 9, "description": "Extra petty"},
    ]
    return {"presets": presets}


@api_router.post("/upload-memes")
async def upload_memes(files: List[UploadFile] = File(...)):
    """Upload multiple meme images and perform OCR with image enhancement"""
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # ENHANCE IMAGE FOR BETTER OCR (handles low-res)
            enhanced_image = preprocess_image_for_ocr(image)
            
            # Perform OCR on enhanced image
            extracted_text = pytesseract.image_to_string(enhanced_image).strip()
            
            # SKIP IMAGES WITH NO TEXT (or very little)
            if len(extracted_text) < 5:  # Less than 5 characters = probably no text
                results.append({
                    "filename": file.filename,
                    "error": "No text detected - image skipped",
                    "status": "skipped"
                })
                continue
            
            # AUTO-CORRECT TEXT
            corrected_text = auto_correct_text(extracted_text)
            
            # EXTRACT KEYWORDS (keeps NSFW words!)
            keywords = extract_keywords(corrected_text, min_length=3, max_keywords=15)
            
            # Skip if no meaningful keywords
            if len(keywords) == 0:
                results.append({
                    "filename": file.filename,
                    "error": "No keywords found - image skipped",
                    "status": "skipped"
                })
                continue
            
            word_count = len(extracted_text.split())
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create meme OCR object
            meme_ocr = MemeOCR(
                filename=file.filename,
                extracted_text=extracted_text,
                corrected_text=corrected_text,
                keywords=keywords,
                word_count=word_count,
                image_data=img_base64
            )
            
            # Save to database
            doc = meme_ocr.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.meme_ocr.insert_one(doc)
            
            results.append({
                "filename": file.filename,
                "extracted_text": extracted_text,
                "corrected_text": corrected_text,
                "keywords": keywords,
                "word_count": word_count,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    successful = len([r for r in results if r['status'] == 'success'])
    skipped = len([r for r in results if r['status'] == 'skipped'])
    
    return {
        "uploaded": len(files), 
        "successful": successful,
        "skipped": skipped,
        "results": results
    }


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
    """Generate new NSFW adult humor memes using keyword algorithm with TONE control"""
    
    # Use custom keywords if provided, otherwise get from database
    if request.keywords and len(request.keywords) > 0:
        top_keywords = request.keywords
    else:
        # Get all extracted data
        ocr_results = await db.meme_ocr.find({}, {"_id": 0}).to_list(1000)
        
        if not ocr_results:
            raise HTTPException(status_code=404, detail="No meme data available. Please upload memes first.")
        
        # COLLECT ALL KEYWORDS (including NSFW)
        all_keywords = []
        for result in ocr_results:
            all_keywords.extend(result.get('keywords', []))
        
        # Get unique keywords with frequency
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common(30)]
    
    # Build keyword patterns
    patterns = build_keyword_patterns(top_keywords)
    
    # Get tone settings (default to 5 if not provided)
    tone = request.tone if request.tone else ToneSettings()
    
    # Get style settings
    style = request.style if request.style else MemeStyle()
    
    # BUILD STYLE-SPECIFIC INSTRUCTIONS
    length_instructions = {
        "short": "1 line only, ultra concise",
        "medium": "1-2 lines max",
        "long": "2-3 lines, more detailed"
    }
    
    format_instructions = {
        "statement": "Make bold statements",
        "question": "Phrase as rhetorical questions",
        "observation": "Start with 'When...', 'That moment when...', 'POV:', etc.",
        "setup_punchline": "Two-part: setup then punchline"
    }
    
    style_desc = f"{length_instructions[style.length]}. {format_instructions[style.format]}."
    
    # BUILD TONE-SPECIFIC PROMPT
    tone_descriptors = []
    if tone.naughty >= 7:
        tone_descriptors.append("EXTRA NAUGHTY and inappropriate")
    elif tone.naughty >= 5:
        tone_descriptors.append("naughty")
    
    if tone.sexy >= 7:
        tone_descriptors.append("EXTRA SEXY with heavy sexual innuendos")
    elif tone.sexy >= 5:
        tone_descriptors.append("sexy and flirty")
    
    if tone.funny >= 7:
        tone_descriptors.append("HILARIOUS and absurd")
    elif tone.funny >= 5:
        tone_descriptors.append("funny")
    
    if tone.rude >= 7:
        tone_descriptors.append("BRUTALLY RUDE and offensive")
    elif tone.rude >= 5:
        tone_descriptors.append("rude and edgy")
    
    tone_description = ", ".join(tone_descriptors) if tone_descriptors else "balanced adult humor"
    
    # Initialize LLM chat with EXPLICIT NSFW instructions + TONE
    api_key = os.environ.get('EMERGENT_LLM_KEY')
    chat = LlmChat(
        api_key=api_key,
        session_id=str(uuid.uuid4()),
        system_message=f"""You are an NSFW meme generator for MISS TITTY SPRINKLES FUCKING FUCKING FACTORY.
Create EXPLICIT adult humor memes with sexual content. 
Use profanity freely (fuck, shit, damn, etc.). 
Include sexual references (sex, boobs, dick, ass, etc.).
Be edgy, raunchy, and hilarious. No filters, no censorship.
TONE: Make it {tone_description}.
This is for adult entertainment only."""
    ).with_model("openai", "gpt-4o")
    
    # Generate memes with NSFW + TONE + STYLE prompt
    keyword_list = ', '.join(top_keywords[:25])
    prompt = f"""Using these NSFW keywords: {keyword_list}

Create {request.count} EXPLICIT NSFW adult humor meme texts.

TONE: {tone_description}
STYLE: {style_desc}

Requirements:
- Use profanity freely (fuck, shit, damn, etc.)
- Include sexual references and innuendos
- Be {tone_description}
- Follow style: {style_desc}
- Hilarious and inappropriate
- Mix keywords creatively

Format response as JSON array with 'text' and 'source_words' (3-5 keywords used):
Example: [{{"text": "When you fuck up but your ass looks good", "source_words": ["fuck", "ass", "good"]}}]

Return ONLY the JSON array."""
    
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
        if not sample_images:
            raise HTTPException(status_code=404, detail="No images available")
        
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
            
            # For now, just use the original image
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            final_img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Determine pattern used
            pattern = "+".join(meme.get('source_words', [])[:3])
            
            generated_meme = GeneratedMeme(
                text=meme['text'],
                image_data=final_img_base64,
                source_words=meme.get('source_words', []),
                keyword_pattern=pattern,
                tone_used=tone
            )
            
            # Save to database
            doc = generated_meme.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            doc['tone_used'] = tone.model_dump()
            await db.generated_memes.insert_one(doc)
            
            generated_memes.append({
                "id": generated_meme.id,
                "text": generated_meme.text,
                "image_data": final_img_base64,
                "source_words": generated_meme.source_words,
                "pattern": pattern,
                "tone": tone.model_dump()
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


@api_router.post("/generate-similar")
async def generate_similar(request: GenerateSimilarRequest):
    """Generate 3 similar variations of an existing meme"""
    
    # Find the original meme
    original = await db.generated_memes.find_one({"id": request.meme_id}, {"_id": 0})
    
    if not original:
        raise HTTPException(status_code=404, detail="Meme not found")
    
    # Use provided tone or original tone
    tone = request.tone if request.tone else ToneSettings(**original.get('tone_used', {}))
    
    # Get source keywords
    source_keywords = original.get('source_words', [])
    
    # Generate 3 similar memes
    gen_request = GenerateMemeRequest(
        count=3,
        tone=tone,
        keywords=source_keywords
    )
    
    result = await generate_new_memes(gen_request)
    return {"similar_memes": result["memes"]}


@api_router.put("/update-text")
async def update_text(request: UpdateTextRequest):
    """Update corrected text for an OCR result"""
    
    # Update in database
    result = await db.meme_ocr.update_one(
        {"id": request.id},
        {"$set": {"corrected_text": request.corrected_text}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Meme not found")
    
    # Re-extract keywords from corrected text
    keywords = extract_keywords(request.corrected_text, min_length=3, max_keywords=15)
    
    await db.meme_ocr.update_one(
        {"id": request.id},
        {"$set": {"keywords": keywords}}
    )
    
    return {"success": True, "updated_keywords": keywords}


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