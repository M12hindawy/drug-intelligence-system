# -*- coding: utf-8 -*-
"""Quick test: Does Groq work for drug RAG?"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

key = os.environ.get('GROQ_API_KEY', '')
print(f"GROQ_API_KEY present: {bool(key)}")
print(f"Key starts with: {key[:10]}..." if key else "NO KEY SET!")

if not key:
    print("\nERROR: Set GROQ_API_KEY first!")
    sys.exit(1)

print("\nTesting Groq connection for drug RAG...")
try:
    from groq import Groq
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a clinical pharmacist."},
            {"role": "user", "content": "What is the adult dose of amoxicillin? One sentence only."}
        ],
        temperature=0.2,
        max_tokens=100
    )
    print(f"\nGroq response: {resp.choices[0].message.content}")
    print("\nGroq OK for drug RAG!")
except Exception as e:
    print(f"\nGroq ERROR: {type(e).__name__}: {e}")
