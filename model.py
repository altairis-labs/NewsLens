"""
model.py -- NewsLens ML Backend
9 categories: Crime, Education, Environment, Health,
              Politics, Science, Sports, Technology, World News
"""
import os, warnings, json
import numpy as np, pandas as pd, joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import preprocess_text, preprocess_series, tokenise_series, tokenise
warnings.filterwarnings("ignore")

MODEL_PATH   = "model.joblib"
DATASET_PATH = "dataset.csv"

# ── TF-IDF ────────────────────────────────────────────────────────────────────
def build_tfidf_word():
    return TfidfVectorizer(analyzer="word", ngram_range=(1,2),
        max_features=60000, sublinear_tf=True, min_df=1)

def build_tfidf_char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
        max_features=30000, sublinear_tf=True, min_df=1)

# ── Doc2Vec ───────────────────────────────────────────────────────────────────
def train_doc2vec(token_lists, vector_size=150, epochs=30):
    tagged = [TaggedDocument(t,[i]) for i,t in enumerate(token_lists)]
    model  = Doc2Vec(vector_size=vector_size, window=5, min_count=2,
                     workers=4, epochs=epochs, dm=1, seed=42)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def get_doc2vec_vectors(d2v, token_lists):
    return np.array([d2v.infer_vector(t, epochs=20) for t in token_lists])

def combine_features(tw, tc, d2v_mat):
    d2v_sparse = csr_matrix(normalize(d2v_mat, norm="l2"))
    return hstack([tw, tc, d2v_sparse], format="csr")

# ── Classifiers ───────────────────────────────────────────────────────────────
def get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(
            C=3.0, max_iter=1000, solver="saga", class_weight="balanced"),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "SVM":                 CalibratedClassifierCV(
            LinearSVC(C=0.8, max_iter=3000, class_weight="balanced"), cv=3),
    }

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, categories):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=categories)
    report = classification_report(y_test, y_pred, labels=categories,
        target_names=categories, zero_division=0, output_dict=True)
    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "report":           report,
    }

# ── Feature importance ────────────────────────────────────────────────────────
def extract_top_features(clf, vectorizer, categories, n=15):
    top = {}
    try:
        feature_names = vectorizer.get_feature_names_out()
        W = len(feature_names)
        if hasattr(clf, "calibrated_classifiers_"):
            coef = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
        elif hasattr(clf, "coef_"):
            coef = clf.coef_
        else:
            return top
        coef_word = coef[:, :W]
        for i, cat in enumerate(categories):
            if i >= coef_word.shape[0]: break
            indices  = np.argsort(coef_word[i])[::-1][:n]
            top[cat] = [feature_names[int(j)] for j in indices]
        print(f"  Feature importance: {len(top)}/{len(categories)} categories")
    except Exception as e:
        print(f"  [warn] extract_top_features: {e}")
    return top

def build_wordcloud_data(df):
    return {cat: df.loc[df["label"]==cat,"text"].str.cat(sep=" ")[:50000]
            for cat in df["label"].unique()}

# ── Classical Predictor ───────────────────────────────────────────────────────
class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: '{model_path}'. Run train_and_save() first.")
        p = joblib.load(model_path)
        self.tfidf_word=p["tfidf_word"]; self.tfidf_char=p["tfidf_char"]
        self.d2v=p["d2v"]; self.models=p["models"]; self.best_name=p["best_name"]
        self.categories=p["categories"]; self.best_accuracy=p["best_accuracy"]
        self.all_metrics=p["all_metrics"]; self.top_features=p["top_features"]
        self.wordcloud_data=p["wordcloud_data"]

    def _vectorise(self, text):
        processed   = preprocess_text(text)
        tokens      = tokenise(text)
        tw          = self.tfidf_word.transform([processed])
        np.random.seed(abs(hash(text)) % (2**31))
        d2v_vec     = self.d2v.infer_vector(tokens, epochs=20).reshape(1,-1)
        np.random.seed(None)
        d2v_sparse  = csr_matrix(normalize(d2v_vec, norm="l2"))
        if self.tfidf_char is not None:
            tc       = self.tfidf_char.transform([processed])
            combined = hstack([tw, tc, d2v_sparse], format="csr")
        else:
            combined = hstack([tw, d2v_sparse], format="csr")
        return tw, combined

    def predict(self, text, model_name=None):
        if not text.strip(): raise ValueError("Input text cannot be empty.")
        nb_vec, combined = self._vectorise(text)
        name = model_name if model_name in self.models else self.best_name
        clf  = self.models[name]
        X    = nb_vec if name == "Naive Bayes" else combined
        pred = clf.predict(X)[0]
        prob = float(max(clf.predict_proba(X)[0]))
        return {"category":pred,"confidence":prob,
                "model_name":name,"model_accuracy":self.all_metrics[name]["accuracy"]}

# ── Transformer Predictor (XLM-RoBERTa) ──────────────────────────────────────
class TransformerPredictor:
    """Loads fine-tuned XLM-RoBERTa. Handles English and Chinese natively."""
    def __init__(self, model_dir):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Transformer model not found: '{model_dir}'.")
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device); self.model.eval()
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f: meta=json.load(f)
            self.categories    = meta["categories"]
            self.id2label      = {int(k):v for k,v in meta["id2label"].items()}
            self.model_metrics = meta.get("metrics",{})
            self.model_name_hf = meta.get("model_name","XLM-RoBERTa")
            self.max_len       = meta.get("max_len",128)
        else:
            cfg=self.model.config; self.id2label={int(k):v for k,v in cfg.id2label.items()}
            self.categories=[self.id2label[i] for i in range(len(self.id2label))]
            self.model_name_hf="XLM-RoBERTa"; self.max_len=128; self.model_metrics={}
        self.all_metrics   = {self.model_name_hf: self.model_metrics}
        self.best_name     = self.model_name_hf
        self.best_accuracy = self.model_metrics.get("accuracy",0.0)
        self.top_features  = {}; self.wordcloud_data={}
        self.models        = {self.model_name_hf: self}
        print(f"Loaded {self.model_name_hf}  acc={self.best_accuracy:.4f}  device={self.device}")

    def predict(self, text, model_name=None):
        import torch
        if not text.strip(): raise ValueError("Input text cannot be empty.")
        enc = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        enc = {k:v.to(self.device) for k,v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))
        return {"category":self.id2label[pred_id],"confidence":float(probs[pred_id]),
                "model_name":self.model_name_hf,"model_accuracy":self.best_accuracy}

# ── Training ──────────────────────────────────────────────────────────────────
def train_and_save(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    print(f"Loading '{dataset_path}' ...")
    df = pd.read_csv(dataset_path)
    # Normalise label name if old version of dataset
    df["label"] = df["label"].replace({"Health & Wellness": "Health"})
    categories  = sorted(df["label"].unique().tolist())
    print(f"  {len(df)} samples | {len(categories)} categories: {categories}")

    X_text   = preprocess_series(df["text"]).tolist()
    X_tokens = tokenise_series(df["text"]).tolist()
    y        = df["label"].tolist()

    X_text_tr,X_text_te,X_tok_tr,X_tok_te,y_train,y_test = train_test_split(
        X_text,X_tokens,y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting TF-IDF ...")
    tfidf_word=build_tfidf_word(); tfidf_char=build_tfidf_char()
    Xw_tr=tfidf_word.fit_transform(X_text_tr); Xw_te=tfidf_word.transform(X_text_te)
    Xc_tr=tfidf_char.fit_transform(X_text_tr); Xc_te=tfidf_char.transform(X_text_te)

    print("Training Doc2Vec ...")
    d2v      = train_doc2vec(X_tok_tr)
    Xd_tr    = get_doc2vec_vectors(d2v, X_tok_tr)
    Xd_te    = get_doc2vec_vectors(d2v, X_tok_te)
    X_tr_all = combine_features(Xw_tr, Xc_tr, Xd_tr)
    X_te_all = combine_features(Xw_te, Xc_te, Xd_te)
    print(f"  Feature matrix: {X_tr_all.shape}  nnz={X_tr_all.nnz:,}")

    classifiers=get_classifiers(); all_metrics={}; trained_models={}; top_features={}
    for name, clf in classifiers.items():
        print(f"Training {name} ...")
        if name=="Naive Bayes":
            clf.fit(Xw_tr, y_train); metrics=evaluate(clf,Xw_te,y_test,categories)
        else:
            clf.fit(X_tr_all, y_train); metrics=evaluate(clf,X_te_all,y_test,categories)
        all_metrics[name]=metrics; trained_models[name]=clf
        print(f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
        if name in ("Logistic Regression","SVM"):
            top_features[name]=extract_top_features(clf,tfidf_word,categories)

    best_name     = max(all_metrics, key=lambda n: all_metrics[n]["accuracy"])
    best_accuracy = all_metrics[best_name]["accuracy"]
    print(f"\nBest: {best_name}  acc={best_accuracy:.4f}")

    payload={"tfidf_word":tfidf_word,"tfidf_char":tfidf_char,"d2v":d2v,
             "models":trained_models,"best_name":best_name,"best_accuracy":best_accuracy,
             "categories":categories,"all_metrics":all_metrics,
             "top_features":top_features,"wordcloud_data":build_wordcloud_data(df)}
    joblib.dump(payload, model_path, compress=3)
    print(f"Saved -> '{model_path}'")
    return payload

if __name__ == "__main__":
    train_and_save()
