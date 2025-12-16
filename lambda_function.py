import json
import boto3
import numpy as np
import io
import re
from typing import List, Dict, Tuple

# ---------- Config ----------
bucket_name = "ecommerce-ai-agent-storage"
embeddings_file = "embeddings.npy"
metadata_file = "metadata.json"

embed_model_id = "amazon.titan-embed-text-v2:0"
text_model_id  = "amazon.titan-text-express-v1"  # إذا عندك :0 استخدمه مثل "amazon.titan-text-express-v1:0"

TOP_K = 12  # عدد المرشحين من الـ RAG قبل التصفية
FINAL_K = 3 # عدد المنتجات النهائية للإجابة

s3 = boto3.client("s3")
runtime = boto3.client("bedrock-runtime")

# Cached in the Lambda execution environment
embeddings = None
products = None


# ---------- HTTP response ----------
def make_response(status_code: int, payload: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


# ---------- Load vectors + metadata ----------
def load_vectors():
    global embeddings, products

    if embeddings is None:
        print("📥 Loading embeddings from S3...")
        npy_data = s3.get_object(Bucket=bucket_name, Key=embeddings_file)["Body"].read()
        embeddings_local = np.load(io.BytesIO(npy_data))
        embeddings_local = embeddings_local.astype(np.float32)
        embeddings_local = np.ascontiguousarray(embeddings_local)
        embeddings_local_norms = np.linalg.norm(embeddings_local, axis=1)
        embeddings_local_norms[embeddings_local_norms == 0] = 1e-12

        # cache both arrays in globals as tuple
        embeddings = (embeddings_local, embeddings_local_norms)
        print(f"✅ Embeddings loaded: {embeddings_local.shape}")

    if products is None:
        print("📄 Loading metadata...")
        meta = s3.get_object(Bucket=bucket_name, Key=metadata_file)["Body"].read()
        products = json.loads(meta)
        print(f"✅ Products loaded: {len(products)} items")


# ---------- Text helpers ----------
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_query(q: str) -> str:
    q = (q or "").strip()
    q = q.translate(ARABIC_NUM_MAP)
    return q

def safe_price(p) -> float:
    try:
        if p is None: 
            return float("nan")
        return float(p)
    except:
        return float("nan")

def extract_budget(query: str) -> float:
    """
    Examples:
      "تحت 200" "اقل من 300" "under 150"
    """
    q = query.lower()
    m = re.search(r"(?:تحت|اقل من|أقل من|under|less than)\s*(\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1))
    # fallback: any number (weak)
    m2 = re.search(r"(\d+(?:\.\d+)?)", q)
    return float(m2.group(1)) if m2 else float("nan")


# ---------- Intent detection ----------
def detect_intent(query: str) -> str:
    q = query.lower()

    if any(w in q for w in ["قارن", "مقارنة", "فرق", "الفرق", "compare", "versus", "vs", "افضل من"]):
        return "COMPARE"

    if any(w in q for w in ["أغلى", "الاغلى", "اغلى", "most expensive", "highest price"]):
        return "MAX_PRICE"

    if any(w in q for w in ["أرخص", "الارخص", "ارخص", "cheapest", "lowest price"]):
        return "MIN_PRICE"

    if any(w in q for w in ["تحت", "اقل من", "أقل من", "under", "less than"]):
        return "BUDGET"

    return "RECOMMEND"


# ---------- Embedding ----------
def embed_query(text: str) -> np.ndarray:
    response = runtime.invoke_model(
        modelId=embed_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    output = json.loads(response["body"].read())
    vec = np.array(output["embedding"], dtype=np.float32)
    return vec


# ---------- RAG retrieval ----------
def cosine_topk(user_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    emb_matrix, emb_norms = embeddings
    u_norm = float(np.linalg.norm(user_vec))
    if u_norm == 0:
        u_norm = 1e-12

    scores = (emb_matrix @ user_vec) / (emb_norms * u_norm)
    # handle any NaNs
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    idx = np.argpartition(scores, -top_k)[-top_k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]


# ---------- Filters ----------
def is_perfume_query(q: str) -> bool:
    ql = q.lower()
    return ("عطر" in ql) or ("perfume" in ql) or ("fragrance" in ql)

def filter_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    q = query.lower()

    # Category hint
    if is_perfume_query(q):
        filtered = []
        for p in candidates:
            cat = (p.get("category") or "").strip()
            tags = " ".join(p.get("tags", []) or [])
            if cat == "عطور" or ("عطر" in tags):
                filtered.append(p)
        candidates = filtered or candidates

    # Budget
    budget = extract_budget(query)
    if not np.isnan(budget):
        under = []
        for p in candidates:
            pr = safe_price(p.get("price"))
            if not np.isnan(pr) and pr <= budget:
                under.append(p)
        candidates = under or candidates

    return candidates


# ---------- Better selection logic ----------
def pick_products(intent: str, query: str, scored: List[Tuple[Dict, float]]) -> List[Dict]:
    """
    scored: list of (product_dict, similarity_score)
    returns selected products (FINAL_K or 1) with a tiny bit of reasoning logic
    """
    # Attach score for later prompt/debug
    enriched = []
    for p, s in scored:
        p2 = dict(p)
        p2["_score"] = round(float(s), 4)
        p2["_price"] = safe_price(p.get("price"))
        enriched.append(p2)

    # filter with rules
    candidates = filter_candidates(query, enriched)
    # keep their scores
    # (candidates already include _score/_price)
    valid_price = [p for p in candidates if not np.isnan(p["_price"])]

    if intent == "MAX_PRICE":
        if valid_price:
            best = max(valid_price, key=lambda x: x["_price"])
            return [best]
        return candidates[:1] if candidates else []

    if intent == "MIN_PRICE":
        if valid_price:
            best = min(valid_price, key=lambda x: x["_price"])
            return [best]
        return candidates[:1] if candidates else []

    if intent == "BUDGET":
        # Under budget: prioritize relevance first, then cheaper among top
        candidates_sorted = sorted(candidates, key=lambda x: (-x["_score"], x["_price"] if not np.isnan(x["_price"]) else 1e9))
        return candidates_sorted[:FINAL_K]

    if intent == "COMPARE":
        # return more items for comparison (up to 4)
        candidates_sorted = sorted(candidates, key=lambda x: -x["_score"])
        return candidates_sorted[:min(4, max(2, FINAL_K+1))]

    # RECOMMEND:
    # combine relevance + mild preference for reasonable price if present
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (
            -x["_score"],
            x["_price"] if not np.isnan(x["_price"]) else 1e9
        )
    )
    return candidates_sorted[:FINAL_K]


# ---------- LLM context + prompting ----------
def compact_product_line(p: Dict) -> str:
    name = p.get("name", "")
    cat = p.get("category", "")
    price = p.get("price", "")
    desc = (p.get("description") or "").strip()
    tags = p.get("tags", [])
    tags_txt = "، ".join(tags[:6]) if isinstance(tags, list) else ""
    return (
        f"- الاسم: {name}\n"
        f"  الفئة: {cat}\n"
        f"  السعر: {price} ريال\n"
        f"  الوسوم: {tags_txt}\n"
        f"  الوصف: {desc}\n"
        f"  (درجة التشابه: {p.get('_score','')})"
    )

def build_context_for_llm(selected: List[Dict]) -> str:
    return "\n\n".join([compact_product_line(p) for p in selected])

def llm_generate_answer(query: str, intent: str, selected: List[Dict]) -> str:
    context_text = build_context_for_llm(selected)

    # Instructions vary by intent
    if intent == "COMPARE":
        task = (
            "قدّم مقارنة واضحة بين المنتجات، واذكر نقاط التشابه والاختلاف (الرائحة/الاستخدام/الطابع/القيمة مقابل السعر). "
            "اختم بتوصية: لمن يناسب كل منتج."
        )
    elif intent == "MAX_PRICE":
        task = "أجب بوضوح عن أغلى خيار مناسب ضمن النتائج، واذكر لماذا قد يستحق السعر، واقترح بديلين أرخص مع سبب."
    elif intent == "MIN_PRICE":
        task = "أجب بوضوح عن أرخص خيار مناسب ضمن النتائج، واذكر تنازلاته إن وجدت، واقترح بديلين أعلى جودة/سعر مع سبب."
    elif intent == "BUDGET":
        task = "اختر أفضل خيارات ضمن الميزانية المذكورة، واذكر سبب الاختيار، مع إبراز المفاضلات."
    else:
        task = "قدّم توصيات مرتبة من الأفضل إلى الأقل، مع أسباب عملية مختصرة ومقارنة سريعة بين الخيارات."

    prompt = f"""
أنت مساعد تسوق إلكتروني ذكي لمتجر سعودي.
مهمتك: استخدام المنتجات المسترجعة من قاعدة البيانات (RAG) فقط. لا تخترع منتجات.

سؤال المستخدم:
{query}

نية السؤال (للاسترشاد):
{intent}

منتجات مسترجعة من قاعدة البيانات:
{context_text}

التعليمات:
- اكتب بالعربية الفصحى الواضحة، جمل ممتازة وسلسة.
- {task}
- لا تذكر "درجة التشابه" للمستخدم.
- إن كانت المعلومات غير كافية، اسأل سؤال توضيحي واحد فقط قبل التوصية.

الجواب:
"""

    response = runtime.invoke_model(
        modelId=text_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 450,
                "temperature": 0.4,
                "topP": 0.9,
            }
        }),
    )
    body = json.loads(response["body"].read())
    return body["results"][0]["outputText"].strip()


# ---------- Main Lambda handler ----------
def lambda_handler(event, context):
    # CORS preflight
    request_method = event.get("requestContext", {}).get("http", {}).get("method", "")
    if request_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": ""
        }

    try:
        load_vectors()

        body = event.get("body", "{}")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return make_response(400, {"error": "Invalid JSON body", "rawBody": body})

        query = normalize_query(body.get("query", ""))
        if not query:
            return make_response(400, {"error": "query required"})

        intent = detect_intent(query)

        # 1) RAG retrieval by embeddings
        user_vec = embed_query(query)
        top_pairs = cosine_topk(user_vec, TOP_K)  # list of (idx, score)

        scored_products = []
        for idx, score in top_pairs:
            if 0 <= idx < len(products):
                scored_products.append((products[idx], score))

        if not scored_products:
            return make_response(200, {"query": query, "answer": "لم أجد أي منتج مناسب للطلب.", "products": []})

        # 2) Select final products using intent-aware logic
        selected = pick_products(intent, query, scored_products)

        if not selected:
            return make_response(200, {"query": query, "answer": "لم أجد أي منتج مناسب للطلب.", "products": []})

        # 3) LLM answer (compare/contrast + good sentences)
        answer = llm_generate_answer(query, intent, selected)

        # Remove internal fields before returning products to UI (optional)
        public_products = []
        for p in selected:
            p_out = dict(p)
            p_out.pop("_score", None)
            p_out.pop("_price", None)
            public_products.append(p_out)

        return make_response(200, {
            "query": query,
            "intent": intent,
            "answer": answer,
            "products": public_products
        })

    except Exception as e:
        print("🔥 ERROR in Lambda:", repr(e))
        return make_response(500, {"error": str(e)})
