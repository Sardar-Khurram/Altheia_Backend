import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
THRESHOLD = 0.8

print("Starting Alethia Backend...")

# Lazy loading variables
transformers_loaded = False
models_loaded = False

def load_models():
    """Lazy load heavy dependencies"""
    global transformers_loaded, models_loaded, tok, model, generator, embedder, nltk, wikipedia, DIGIT_RE, PROPER_RE
    
    if not transformers_loaded:
        print("Loading transformers and dependencies...")
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            import wikipedia
            from wikipedia.exceptions import DisambiguationError, PageError
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from sentence_transformers import SentenceTransformer, util
            
            nltk.download('punkt', quiet=True)
            wikipedia.set_lang("en")
            
            # Regex patterns for claim extraction
            DIGIT_RE = re.compile(r"\d")
            PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b")
            
            transformers_loaded = True
            print("Dependencies loaded successfully!")
        except ImportError as e:
            print(f"Error loading dependencies: {e}")
            return False
    
    if not models_loaded and transformers_loaded:
        print("Loading ML models...")
        try:
            # GPT-2 setup
            llm_name = "gpt2"
            tok = AutoTokenizer.from_pretrained(llm_name)
            model = AutoModelForCausalLM.from_pretrained(llm_name)

            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

            device = 0 if torch.cuda.is_available() else -1
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                device=device
            )

            # Sentence transformer for embeddings
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", 
                                          device="cuda" if torch.cuda.is_available() else "cpu")

            models_loaded = True
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    return transformers_loaded and models_loaded

def get_draft_answer(user_question: str) -> str:
    """Generate answer using GPT-2 or fallback"""
    if not models_loaded:
        # Fallback response for demo
        return f"This is a demo response for: {user_question}. GPT-2 would provide a more detailed answer here."
    
    prompt = f"Q: {user_question}\nA:"
    out = generator(
        prompt,
        max_new_tokens=160,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tok.eos_token_id
    )[0]["generated_text"]

    answer = out[len(prompt):].strip()
    cut = answer.split("\nQ:")[0].strip()
    return cut if cut else answer

def extract_claims(text: str):
    """Extract factual claims from text"""
    if not transformers_loaded:
        # Simple fallback claim extraction
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) >= 6]
        return sentences if sentences else [text.strip()]
    
    from nltk.tokenize import sent_tokenize
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    claims = []
    for s in sentences:
        if len(s.split()) >= 6 and (DIGIT_RE.search(s) or PROPER_RE.search(s)):
            claims.append(s)
    if not claims and text.strip():
        claims = [text.strip()]
    return claims

def wiki_best_match(query: str, k_titles: int = 5, summary_sents: int = 3):
    """Find best Wikipedia match for query"""
    if not transformers_loaded:
        # Fallback demo data
        return f"Demo Wikipedia Article for {query}", "https://en.wikipedia.org/wiki/Demo", f"This is a demo Wikipedia summary for the query: {query}. In a real implementation, this would fetch actual Wikipedia content.", 0.7
    
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError
    from sentence_transformers import util
    
    try:
        titles = wikipedia.search(query)
    except Exception:
        titles = []

    if not titles:
        return None, None, None, 0.0

    q_emb = embedder.encode([query], convert_to_tensor=True)
    best = (0.0, None, None, None)

    for title in titles[:k_titles]:
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            summ = wikipedia.summary(title, sentences=summary_sents, auto_suggest=False, redirect=True)
        except DisambiguationError as e:
            tried_any = False
            for opt in e.options[:3]:
                try:
                    page = wikipedia.page(opt, auto_suggest=False, redirect=True)
                    summ = wikipedia.summary(opt, sentences=summary_sents, auto_suggest=False, redirect=True)
                    tried_any = True
                except Exception:
                    continue
                if not summ or len(summ.split()) < 15:
                    continue
                s_emb = embedder.encode([summ], convert_to_tensor=True)
                score = float(util.cos_sim(q_emb, s_emb)[0][0])
                if score > best[0]:
                    best = (score, page.title, page.url, summ)
            if not tried_any:
                continue
        except Exception:
            continue

        if not summ or len(summ.split()) < 15:
            continue

        s_emb = embedder.encode([summ], convert_to_tensor=True)
        score = float(util.cos_sim(q_emb, s_emb)[0][0])
        if score > best[0]:
            best = (score, page.title, page.url, summ)

    return best[1], best[2], best[3], best[0]

def verify_claims_with_wikipedia(claims):
    """Verify claims against Wikipedia"""
    rows = []
    for claim in claims:
        title, url, snippet, _ = wiki_best_match(claim, k_titles=5, summary_sents=3)
        if title is None:
            score = 0.0
            snippet = ""
            url = None
        else:
            if models_loaded:
                from sentence_transformers import util
                c_emb = embedder.encode([claim], convert_to_tensor=True)
                s_emb = embedder.encode([snippet], convert_to_tensor=True)
                score = float(util.cos_sim(c_emb, s_emb)[0][0])
            else:
                # Simple demo scoring based on keyword overlap
                claim_words = set(claim.lower().split())
                snippet_words = set(snippet.lower().split())
                common_words = claim_words.intersection(snippet_words)
                score = len(common_words) / max(len(claim_words), 1) if claim_words else 0.5

        rows.append({
            "claim": claim,
            "similarity": round(score, 3),
            "source_title": title if title else "",
            "source_url": url if url else "",
            "snippet": snippet[:300].replace("\n", " ") if snippet else ""
        })

    avg = sum(row["similarity"] for row in rows) / len(rows) if rows else 0.0
    return rows, avg

@app.route("/", methods=["GET"])
def home():
    return "Alethia Backend is Running!"

@app.route("/api/verify", methods=["POST"])
def verify_question():
    try:
        # Try to load models if not already loaded
        if not models_loaded:
            load_models()
        
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Question is required"}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Generate draft answer
        draft = get_draft_answer(question)
        
        # Extract claims
        claims = extract_claims(draft)
        if not claims:
            claims = [draft]

        # Verify claims
        verification_results, avg_score = verify_claims_with_wikipedia(claims)
        
        # Make decision
        is_verified = avg_score >= THRESHOLD
        
        response = {
            "question": question,
            "gpt2_response": draft,
            "claims": verification_results,
            "average_score": round(avg_score, 3),
            "threshold": THRESHOLD,
            "is_verified": is_verified,
            "final_answer": draft if is_verified else "I don't exactly know the answer. Can you clarify your prompt please?",
            "demo_mode": not models_loaded
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=True)