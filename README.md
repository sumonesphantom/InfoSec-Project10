# LSTM-Driven Phishing Detection for Enterprise Email Security

A Bidirectional LSTM model with an attention mechanism to classify phishing emails in enterprise email systems. The model analyzes email content (subject lines, message bodies, and structure) to detect phishing patterns that mimic normal workplace communication.

## Project Structure

```
InfoSec-Project10/
├── Dataset/                    # Phishing email dataset (82,486 emails)
│   ├── phishing_email.csv      # Main combined dataset (used for training)
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   └── SpamAssasin.csv
├── src/
│   ├── preprocess.py           # Text cleaning, tokenization, sequence preparation
│   ├── model.py                # BiLSTM + Attention model architecture
│   ├── train.py                # Training pipeline with checkpointing
│   ├── evaluate.py             # Standalone evaluation (no retraining needed)
│   ├── explain.py              # LIME/SHAP explainability analysis
│   └── api.py                  # FastAPI REST API for deployment
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── models/                     # Saved model weights and tokenizer
│   ├── best_model.keras        # Best trained model
│   ├── tokenizer.pkl           # Fitted tokenizer
│   ├── X_test.npy              # Test features (for evaluation)
│   └── y_test.npy              # Test labels (for evaluation)
├── results/                    # Evaluation results and plots
│   ├── classification_report.txt
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
├── requirements.txt
├── Dockerfile
└── README.md
```

## Model Architecture

```
Input (200 tokens) -> Embedding (128d) -> SpatialDropout1D
    -> Bidirectional LSTM (128 units per direction, returns sequences)
    -> Attention Layer (128 units, learns word importance weights)
    -> BatchNormalization -> Dense (64, ReLU) -> Dropout (0.3)
    -> Dense (32, ReLU) -> Dense (1, Sigmoid)
```

**Key design choices:**
- **Bidirectional LSTM**: Captures context from both directions in email text
- **Attention Mechanism**: Learns which words/phrases are most indicative of phishing, improving interpretability
- **Regularization**: SpatialDropout1D, Dropout, L2 regularization, and BatchNormalization to prevent overfitting

**Total Parameters**: 6,715,777 (25.62 MB)

## Dataset

The [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data) contains:
- **82,486 emails** (42,891 phishing / 39,595 legitimate)
- **Columns**: `text_combined` (email text), `label` (0 = legitimate, 1 = phishing)
- Sources: CEAS_08, Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin

## Setup

### Prerequisites
- Python 3.10+
- pip

### Download Dataset

The dataset is not included in this repository due to file size limits. Download it from Kaggle:

1. Go to the [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data) on Kaggle
2. Download and extract the CSV files into the `Dataset/` directory

### Install Dependencies

```bash
pip install -r requirements.txt
```

Download NLTK data (run once):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

## Usage

### 1. Training

```bash
python src/train.py
```

- Preprocesses all 82K emails (cleaning, tokenization, lemmatization)
- Splits data: 70% train / 10% validation / 20% test
- Trains BiLSTM + Attention model for up to 20 epochs with early stopping
- **Auto-saves** the best model based on validation loss
- **Skips retraining** if an existing model with >=95% validation accuracy is found
- Saves accuracy-milestone checkpoints in `models/`

### 2. Evaluation (without retraining)

```bash
python src/evaluate.py
```

Loads the saved model and generates:
- Classification report (precision, recall, F1-score)
- Confusion matrix plot
- ROC curve plot
- Precision-recall curve plot

### 3. Explainability Analysis

```bash
python src/explain.py
```

Generates:
- **LIME explanations**: Shows which words contribute to phishing/legitimate classification
- **Attention visualizations**: Highlights the most attended words by the model
- Results saved to `results/explanations/`

### 4. Exploratory Data Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/eda.ipynb
```

### 5. REST API Deployment

```bash
python src/api.py
```

Or with uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /` - API status
- `GET /health` - Health check (model loaded status)
- `POST /predict` - Classify an email

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Dear user, your account has been compromised. Click here to verify your identity immediately."}'
```

**Example response:**
```json
{
  "prediction": "Phishing",
  "confidence": 0.9823,
  "phishing_probability": 0.9823,
  "top_attention_words": [
    {"word": "compromised", "attention": 0.152},
    {"word": "verify", "attention": 0.134},
    {"word": "immediately", "attention": 0.098}
  ]
}
```

### 6. Docker Deployment

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

## Results

Test set evaluation (16,497 emails):

| Metric    | Legitimate | Phishing | Overall |
|-----------|-----------|----------|---------|
| Precision | 0.9822    | 0.9912   | 0.9869  |
| Recall    | 0.9905    | 0.9834   | 0.9868  |
| F1-Score  | 0.9864    | 0.9873   | 0.9868  |
| **Accuracy** |        |          | **0.9868** |

## Text Preprocessing Pipeline

1. **Lowercasing** all text
2. **URL replacement** with `url` token
3. **Email address replacement** with `email` token
4. **HTML tag removal**
5. **Special character and digit removal**
6. **Tokenization** using NLTK
7. **Stop word removal**
8. **Lemmatization** using WordNet
9. **Sequence padding/truncation** to 200 tokens

## Tools and Technologies

- **Python** - Primary programming language
- **TensorFlow / Keras** - Deep learning framework for BiLSTM model
- **NLTK** - Natural language processing and text preprocessing
- **Scikit-learn** - Evaluation metrics and data splitting
- **Pandas / NumPy** - Data manipulation
- **LIME / SHAP** - Explainable AI for model interpretability
- **FastAPI** - REST API for model serving
- **Docker** - Containerization for deployment
- **Matplotlib / Seaborn** - Visualization

## References

- Li et al. (2022), "LSTM Based Phishing Detection for Big Email Data," IEEE Trans. Big Data
- Adebowale et al. (2023), "Intelligent phishing detection using deep learning algorithms," JEIM
- Peng et al. (2021), "A phishing email detection method based on attention mechanism," IEEE Access
- Fang et al. (2019), "Phishing email detection using improved RCNN model," IEEE Access
- Do et al. (2022), "Deep Learning for Phishing Detection: Taxonomy, Current Challenges and Future Directions," IEEE Access
- Sun et al. (2021), "Federated Phish Bowl: LSTM-Based Decentralized Phishing Email Detection," arXiv
