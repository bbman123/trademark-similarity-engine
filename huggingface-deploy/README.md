---
title: Trademark Similarity API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Trademark Similarity Engine API

AI-powered trademark similarity detection using Hybrid CNN+SVM model.

## Features
- Real-time similarity scoring
- Multi-language support (English, Hausa, Yoruba)
- Visual, phonetic, and semantic analysis
- REST API with interactive documentation

## Usage

Once deployed, visit:
- `/docs` - Interactive API documentation
- `/health` - Health check endpoint
- `/similarity-check` - Main similarity endpoint

## Example Request

```bash
curl -X POST "https://huggingface.co/spaces/buhari123/trademark-similarity-api/similarity-check" \
  -H "Content-Type: application/json" \
  -d '{"mark1": "SuperCoffee", "mark2": "Super Coffee"}'
```
