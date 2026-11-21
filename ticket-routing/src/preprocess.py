"""Text cleaning shared by training and the Streamlit app (keeps both identical)."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop URLs/emails/digits/punctuation, collapse whitespace."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
