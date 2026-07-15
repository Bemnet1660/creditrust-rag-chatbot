import re

def clean_complaint_text(text):
    """
    Clean raw complaint narrative:
    - Lowercase
    - Remove special characters/numbers
    - Remove common boilerplate phrases
    - Strip extra spaces
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove boilerplate
    boilerplate_phrases = [
        "i am writing to file a complaint",
        "i would like to file a complaint",
        "i am writing to complain about"
    ]
    for phrase in boilerplate_phrases:
        text = text.replace(phrase, "")
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
