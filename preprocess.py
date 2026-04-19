"""
preprocess.py -- NewsLens Text Preprocessing
English : NLTK POS-aware lemmatization
Chinese : jieba segmentation
"""
import re, string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

for _pkg, _kind in [
    ("wordnet",                        "corpora"),
    ("omw-1.4",                        "corpora"),
    ("stopwords",                      "corpora"),
    ("averaged_perceptron_tagger",     "taggers"),
    ("averaged_perceptron_tagger_eng", "taggers"),
]:
    try:
        nltk.data.find(f"{_kind}/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

try:
    import jieba; jieba.setLogLevel(60); JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()

_BASE_SW = set(nltk.corpus.stopwords.words("english"))
_EXTRA_SW = {
    "said","say","says","told","new","one","two","three","also","would","could",
    "may","will","get","got","go","going","gone","make","made","take","took",
    "taken","use","used","using","like","just","even","still","back","way",
    "year","years","time","times","day","days","week","weeks","month","months",
    "people","person","man","men","woman","women","first","last","good","great",
    "big","high","old","small","little","want","need","know","think","see",
    "look","come","came","report","reports","reported","according","official",
    "officials","percent","number","million","billion","hundred","thousand",
    "however","though","although","since","while","after","before","include",
    "including","included","amid","despite","without",
}
STOPWORDS = _BASE_SW | _EXTRA_SW

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_CJK_RE    = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

def _has_cjk(text): return bool(_CJK_RE.search(text))
def _jieba_tokens(text):
    if not JIEBA_AVAILABLE: return []
    return [t.strip() for t in jieba.cut(text) if len(t.strip()) > 1]
def extract_entities(text):
    return [e.lower().replace(" ","_") for e in _ENTITY_RE.findall(text)]
def _wn_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN
def _lemmatize(tokens):
    try:
        tagged = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(w, _wn_pos(p)) for w, p in tagged]
    except Exception:
        return [lemmatizer.lemmatize(t) for t in tokens]
def _clean(text):
    text = re.sub(r"https?://\S+|www\.\S+","",text)
    text = re.sub(r"<[^>]+>","",text)
    text = text.lower()
    text = re.sub(r"\d+","",text)
    text = text.translate(str.maketrans("","",string.punctuation))
    return re.sub(r"\s+"," ",text).strip()

def tokenise(text, use_stem=False):
    if not isinstance(text,str): text=str(text)
    if not text.strip(): return []
    if _has_cjk(text):
        cjk=_jieba_tokens(text); latin=_CJK_RE.sub(" ",text)
        c=_clean(latin); toks=[t for t in re.findall(r"\b[a-z]{3,}\b",c) if t not in STOPWORDS]
        toks=_lemmatize(toks)
        if use_stem: toks=[stemmer.stem(t) for t in toks]
        return toks+cjk
    c=_clean(text); toks=[t for t in re.findall(r"\b[a-z]{3,}\b",c) if t not in STOPWORDS]
    toks=_lemmatize(toks)
    if use_stem: toks=[stemmer.stem(t) for t in toks]
    return toks

def preprocess_text(text, use_stem=False):
    if not isinstance(text,str): text=str(text)
    if not text.strip(): return ""
    entities=extract_entities(text)
    if _has_cjk(text):
        cjk=_jieba_tokens(text); latin=_CJK_RE.sub(" ",text)
        c=_clean(latin); toks=[t for t in re.findall(r"\b[a-z]{3,}\b",c) if t not in STOPWORDS]
        toks=_lemmatize(toks)
        if use_stem: toks=[stemmer.stem(t) for t in toks]
        return " ".join(toks+cjk+entities)
    c=_clean(text); toks=[t for t in re.findall(r"\b[a-z]{3,}\b",c) if t not in STOPWORDS]
    toks=_lemmatize(toks)
    if use_stem: toks=[stemmer.stem(t) for t in toks]
    toks+=entities
    return " ".join(toks)

def preprocess_series(series, use_stem=False):
    return series.apply(lambda t: preprocess_text(t, use_stem=use_stem))
def tokenise_series(series, use_stem=False):
    return series.apply(lambda t: tokenise(t, use_stem=use_stem))
