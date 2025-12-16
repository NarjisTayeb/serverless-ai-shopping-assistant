# 🛍️ Serverless AI Shopping Assistant (AWS Bedrock)

A scalable serverless AI backend using **AWS Lambda**, **Amazon Bedrock** (Titan embeddings & text), and **API Gateway**, delivering real-time Arabic responses via a **RAG-based pipeline**. Built with semantic search and intent detection for intelligent product recommendations.

**Project by:** [Narjis Bin Tayeb](https://www.linkedin.com/in/narjis-tayeb) | Applied Data Scientist specializing in LLMs, RAG systems, and Cloud Solutions

## 🌟 Features

- **🔍 Semantic Search**: Find products using natural language in Arabic or English
- **🤖 Intent Detection**: Automatically detects user intent (compare, budget, recommendations, etc.)
- **💰 Smart Filtering**: Filters by budget, category, and price range
- **📊 Product Comparison**: Compare multiple products with detailed analysis
- **⚡ Fast Response**: Pre-computed embeddings for millisecond-level search
- **🌐 Bilingual Support**: Works with Arabic and English queries

## 🏗️ Architecture

```
User Query → API Gateway → Lambda Function → Bedrock (Titan)
                              ↓
                          S3 Bucket
                    (Embeddings + Metadata)
```

### Components:

1. **Pre-processing (Colab)**: Generate embeddings for 300+ products
2. **Storage (S3)**: Store embeddings and product metadata
3. **Runtime (Lambda)**: Handle queries, retrieve similar products, generate responses
4. **AI Models (Bedrock)**:
   - `amazon.titan-embed-text-v2:0` - Generate 1536-dim embeddings
   - `amazon.titan-text-express-v1` - Generate natural language responses

## 🚀 Quick Start

### Prerequisites

- AWS Account with Bedrock access
- Python 3.9+
- AWS CLI configured
- S3 bucket created

### 1. Generate Embeddings (One-time setup)

```python
# Run in Google Colab or local environment
import json, numpy as np, boto3, time

bucket_name = "ecommerce-ai-agent-storage"
products_file = "products_300.json"
model_id = "amazon.titan-embed-text-v2:0"

# Load products
with open(products_file, "r", encoding="utf-8") as f:
    products = json.load(f)

texts = [item["name"] + " " + item.get("description","") for item in products]

# Generate embeddings
bedrock = boto3.client("bedrock-runtime")
vectors = []

for i, text in enumerate(texts):
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    vec = json.loads(response["body"].read())["embedding"]
    vectors.append(vec)
    time.sleep(0.1)  # Rate limiting

# Save files
embeddings = np.array(vectors)
np.save("embeddings.npy", embeddings)

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(products, f, ensure_ascii=False)

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file('embeddings.npy', bucket_name, 'embeddings.npy')
s3.upload_file('metadata.json', bucket_name, 'metadata.json')
```

### 2. Deploy Lambda Function

1. **Create Lambda function** (Python 3.9+)
2. **Copy the Lambda code** from `lambda_function.py`
3. **Configure**:
   - Memory: 512 MB
   - Timeout: 60 seconds
   - Runtime: Python 3.9

4. **Add IAM permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::ecommerce-ai-agent-storage/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*:*:foundation-model/*"
    }
  ]
}
```

### 3. Set Up API Gateway

1. Create HTTP API
2. Add POST route → Lambda integration
3. Enable CORS
4. Deploy API

### 4. Test It!

```bash
curl -X POST https://your-api-id.execute-api.region.amazonaws.com/prod \
  -H "Content-Type: application/json" \
  -d '{"query": "عطر رجالي فخم تحت 500 ريال"}'
```

## 📝 API Reference

### Request

```json
POST /
{
  "query": "string"  // User's search query in Arabic or English
}
```

### Response

```json
{
  "query": "عطر رجالي فخم تحت 500 ريال",
  "intent": "BUDGET",
  "answer": "إليك أفضل الخيارات ضمن ميزانيتك...",
  "products": [
    {
      "id": "12345",
      "name": "عطر دولتشي آند غابانا",
      "category": "عطور",
      "price": "450",
      "description": "...",
      "tags": ["رجالي", "فخم", "خشبي"]
    }
  ]
}
```

## 🎯 Supported Intents

| Intent | Example Queries | Behavior |
|--------|----------------|----------|
| **RECOMMEND** | "عطر رجالي" | Returns top 3 relevant products |
| **COMPARE** | "قارن بين عطر A و B" | Compares multiple products |
| **BUDGET** | "تحت 500 ريال" | Filters by price ceiling |
| **MAX_PRICE** | "أغلى عطر" | Returns most expensive match |
| **MIN_PRICE** | "أرخص خيار" | Returns cheapest match |

## 🧠 How It Works

### 1. **Embedding Generation** (Preprocessing)
```
Product Text → Titan Embed V2 → 1536-dim Vector → S3
```

### 2. **Query Processing** (Runtime)
```
User Query → Titan Embed V2 → Query Vector
                                    ↓
            Cosine Similarity with Product Vectors
                                    ↓
                        Top 12 Candidates Retrieved
                                    ↓
                    Intent-Based Filtering + Ranking
                                    ↓
                          Final 3 Products Selected
                                    ↓
            Titan Text Express → Natural Language Response
```

### 3. **Smart Selection Logic**

```python
# Example: Budget filtering
budget = extract_budget("تحت 500 ريال")  # → 500.0
filtered = [p for p in candidates if p["price"] <= budget]

# Example: Category filtering  
if is_perfume_query(query):
    filtered = [p for p in candidates if p["category"] == "عطور"]
```

## 📊 Performance

- **Cold Start**: ~3-5 seconds (loading embeddings from S3)
- **Warm Request**: ~1-2 seconds (embedding query + LLM generation)
- **Search Accuracy**: Semantic similarity using cosine distance
- **Cost**: ~$0.001 per query (Bedrock Titan pricing)

## 🔧 Configuration

Key parameters in `lambda_function.py`:

```python
TOP_K = 12       # Candidates from RAG before filtering
FINAL_K = 3      # Final products in response
embed_model_id = "amazon.titan-embed-text-v2:0"
text_model_id = "amazon.titan-text-express-v1"
```

## 📁 Project Structure

```
.
├── colab_embedding_generation.ipynb  # One-time embedding generation
├── lambda_function.py                # Main Lambda handler
├── products_300.json                 # Product catalog (not included)
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## 🛠️ Tech Stack

- **AWS Lambda**: Serverless compute
- **AWS Bedrock**: AI model access (Titan Embed V2 + Text Express)
- **AWS S3**: Storage for embeddings and metadata
- **AWS API Gateway**: HTTP endpoint
- **NumPy**: Vector operations and cosine similarity
- **Python 3.9**: Runtime environment

## 🌍 Use Cases

- **E-commerce Search**: Natural language product search
- **Recommendation Engine**: Context-aware suggestions
- **Price Comparison**: Budget-conscious shopping
- **Multilingual Support**: Arabic/English queries

## 🚧 Future Enhancements

- [ ] Add user preference learning
- [ ] Implement conversation history
- [ ] Add image-based search
- [ ] Support for filters (brand, rating, availability)
- [ ] Real-time inventory updates
- [ ] A/B testing for ranking algorithms

## 📄 License

MIT License - Feel free to use this project for learning or commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please give it a star!**
