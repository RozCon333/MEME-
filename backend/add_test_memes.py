#!/usr/bin/env python3
"""
Quick script to add test meme data to MongoDB
Run: python add_test_memes.py
"""

import asyncio
import motor.motor_asyncio
from datetime import datetime, timezone

async def insert_test_data():
    # Connect to MongoDB
    client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017/")
    db = client["meme_factory"]
    
    # Test meme data with actual funny meme text
    test_memes = [
        {
            "filename": "test1.jpg",
            "extracted_text": "When you finally understand the assignment",
            "corrected_text": "When you finally understand the assignment",
            "keywords": ["assignment", "understand", "finally", "when", "you"],
            "word_count": 6,
            "image_data": "base64encodeddata",
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "filename": "test2.jpg",
            "extracted_text": "My code works I have no idea how",
            "corrected_text": "My code works I have no idea how",
            "keywords": ["code", "works", "idea", "how", "my"],
            "word_count": 6,
            "image_data": "base64encodeddata",
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "filename": "test3.jpg",
            "extracted_text": "POV you forgot to save your file",
            "corrected_text": "POV you forgot to save your file",
            "keywords": ["POV", "forgot", "save", "file", "you"],
            "word_count": 6,
            "image_data": "base64encodeddata",
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "filename": "test4.jpg",
            "extracted_text": "Pretend it's a feature not a bug",
            "corrected_text": "Pretend it's a feature not a bug",
            "keywords": ["feature", "bug", "pretend", "not"],
            "word_count": 6,
            "image_data": "base64encodeddata",
            "timestamp": datetime.now(timezone.utc)
        },
        {
            "filename": "test5.jpg",
            "extracted_text": "Googling the error instead of fixing it",
            "corrected_text": "Googling the error instead of fixing it",
            "keywords": ["Googling", "error", "fixing", "instead"],
            "word_count": 6,
            "image_data": "base64encodeddata",
            "timestamp": datetime.now(timezone.utc)
        }
    ]
    
    try:
        # Clear existing data
        result = await db.meme_ocr.delete_many({})
        print(f"✓ Cleared {result.deleted_count} existing meme OCR records")
        
        # Insert test data
        result = await db.meme_ocr.insert_many(test_memes)
        print(f"✓ Inserted {len(result.inserted_ids)} test memes")
        
        # Verify insertion
        count = await db.meme_ocr.count_documents({})
        print(f"✓ Total memes in database: {count}")
        
        # Show sample keywords
        sample = await db.meme_ocr.find_one({})
        if sample:
            print(f"\nSample meme: {sample['filename']}")
            print(f"  Text: {sample['extracted_text']}")
            print(f"  Keywords: {sample['keywords']}")
        
        print("\n✅ Test data ready! Now you can generate memes with /api/generate-new-memes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(insert_test_data())
