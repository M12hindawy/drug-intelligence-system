# 💊 Drug Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-API-green?style=for-the-badge&logo=flask)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)

**A full-stack AI-powered pharmaceutical intelligence platform combining 6 deep learning architectures, a RAG-based Q&A engine, and real Egyptian drug market data.**

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Drug Intelligence System                      │
├────────────────┬────────────────┬───────────────────────────────┤
│  Data Layer    │   ML Layer     │      Application Layer        │
│                │                │                               │
│ • Web Scraper  │ • DNN          │ • Flask REST API              │
│   (dwaprices   │ • LSTM         │ • Streamlit Dashboard         │
│    chefaa.com) │ • BiLSTM       │ • RAG Q&A Engine              │
│ • Egyptian Drug│ • CNN1D        │ • Hallucination Detector      │
│   Database     │ • BERT         │ • Clinical Decision Engine    │
│ • ChromaDB     │ • Ensemble     │ • Docker Deployment           │
│   Vector Store │   Voting       │                               │
└────────────────┴────────────────┴───────────────────────────────┘
```

## ✨ Key Features

### 🔬 6 Deep Learning Architectures
| Model | Type | Use Case |
|-------|------|----------|
| DNN | Feedforward Neural Network | Baseline drug classification |
| LSTM | Long Short-Term Memory | Sequential drug name pattern recognition |
| **BiLSTM** | Bidirectional LSTM | Context-aware drug name disambiguation |
| **CNN1D** | 1D Convolutional Neural Network | Local pattern extraction in drug sequences |
| **BERT** | Transformer-based | Semantic understanding of pharmaceutical text |
| Ensemble | Weighted Voting | Production-grade combined prediction |

### 🧠 RAG-Powered Pharmaceutical Q&A
- **ChromaDB** vector store with real Egyptian pharmaceutical data
- **Hallucination Detection Module** — flags uncertain or contradictory responses
- Semantic search over drug interactions, contraindications, dosing guidelines
- Context-aware retrieval with source attribution

### 🌐 Real Egyptian Market Data
- Web-scraped from **dwaprices.com** and **chefaa.com**
- Real drug prices, availability, and generic alternatives
- Updated database with Egyptian pharmacy market coverage

### 🏥 Clinical Decision Support
- Drug interaction checker
- Contraindication alerts
- Dosing recommendations by patient profile
- Arabic-English bilingual interface

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Docker (optional, recommended)
```

### Installation
```bash
git clone https://github.com/M12hindawy/drug-intelligence-system.git
cd drug-intelligence-system
pip install -r requirements.txt
```

### Run with Docker
```bash
docker build -t drug-intelligence .
docker run -p 5000:5000 drug-intelligence
```

### Run Locally
```bash
python app.py
```
Then open `http://localhost:5000`

## 📁 Project Structure

```
drug-intelligence-system/
├── app.py                    # Flask web application entry point
├── inference.py              # Model inference pipeline
├── drug_rag.py               # RAG engine (ChromaDB + LLM)
├── hallucination_detector.py # Response verification module
├── clinical_decision.py      # Clinical decision support engine
├── drug_interactions.py      # Drug-drug interaction checker
├── drug_knowledge_base.py    # Knowledge base management
├── database.py               # Database models and ORM
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
├── Procfile                  # Process configuration (Railway/Heroku)
└── templates/                # Flask HTML templates
```

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, Flask, SQLAlchemy |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **NLP Models** | BERT (bert-base-uncased), BiLSTM, CNN1D |
| **Vector Database** | ChromaDB |
| **RAG Framework** | Custom RAG pipeline |
| **Data Collection** | BeautifulSoup, Requests (web scraping) |
| **Deployment** | Docker, Railway |
| **Database** | SQLite (dev) / PostgreSQL (prod) |

## 📊 Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| DNN | Baseline | Feedforward, fast inference |
| LSTM | +3% vs DNN | Sequential patterns |
| BiLSTM | +5% vs DNN | Bidirectional context |
| CNN1D | +4% vs DNN | Local feature extraction |
| **BERT** | **Best single** | Semantic understanding |
| **Ensemble** | **Best overall** | Production model |

## 🌍 Use Cases

1. **Pharmacists** — Quick drug interaction checking and dosing verification
2. **Healthcare providers** — Clinical decision support
3. **Patients** — Drug information Q&A in Arabic and English
4. **Researchers** — Pharmaceutical data analysis

## 👤 Author

**Mohamed Ashraf Abdelhamed**  
📍 Giza, Egypt | AI & Data Engineering Student  
🔗 [LinkedIn](https://linkedin.com/in/mohamed-ashraf-b7b5a72b6/) | [GitHub](https://github.com/M12hindawy) | [Portfolio](https://mohamedhendawy.netlify.app)

*Built as part of GeoAI & Deep Learning specialization coursework and personal research.*

## 📄 License

MIT License — See [LICENSE](LICENSE) file.
