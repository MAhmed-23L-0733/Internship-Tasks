📰 News Classification with BERT

A machine learning project that fine-tunes BERT (Bidirectional Encoder Representations from Transformers) for automated news article classification using the AG News dataset.

🎯 Project Overview

This project implements a state-of-the-art text classification system that can automatically categorize news articles into four distinct categories:

- **World News** 🌍
- **Sports** ⚽
- **Business** 💼
- **Science/Technology** 🔬

🔧 Technical Stack

- **Model**: BERT-base-uncased (Google's pre-trained transformer)
- **Framework**: Hugging Face Transformers
- **Dataset**: AG News (120,000 training samples, 7,600 test samples)
- **Metrics**: Accuracy and weighted F1-score
- **Training**: Fine-tuning with custom training arguments

📊 Dataset Information

The AG News dataset contains news articles with four categories:

- Total training samples: 120,000
- Total test samples: 7,600
- Text preprocessing: Tokenization with 128 max sequence length
- Label encoding: Integer labels (0-3) for four categories

🚀 Getting Started

Prerequisites

```bash
pip install transformers
pip install datasets
pip install torch
pip install scikit-learn
pip install numpy
```

Running the Project

1. **Open the Jupyter notebook**: `Task1_P2.ipynb`
2. **Run all cells sequentially** to:
   - Load and explore the AG News dataset
   - Tokenize text data using BERT tokenizer
   - Configure training parameters
   - Fine-tune the BERT model
   - Evaluate model performance
   - Save the trained model

Training Configuration

```python
- Learning Rate: 2e-5
- Batch Size: 16 (train/eval)
- Epochs: 3
- Weight Decay: 0.01
- Max Sequence Length: 128
- Evaluation Strategy: Per epoch
```

📈 Model Performance

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score for multi-class performance
- **Best Model Selection**: Based on accuracy metric

💾 Model Output

After training, the model and tokenizer are saved to:

```
./news_classifier/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

🔍 Key Features

- **Pre-trained BERT**: Leverages Google's BERT-base-uncased model
- **Fine-tuning**: Custom fine-tuning for news classification task
- **Efficient Training**: Optimized training arguments for best performance
- **Model Persistence**: Saves trained model for future inference
- **Comprehensive Evaluation**: Multiple metrics for performance assessment

📝 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

Load the trained model
model = AutoModelForSequenceClassification.from_pretrained('./news_classifier')
tokenizer = AutoTokenizer.from_pretrained('./news_classifier')

Classify a news article
text = "Apple Inc. reported strong quarterly earnings..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax().item()
```

🏗️ Project Structure

```
📁 Internship Tasks/
├── 📁 notebooks/
│   └── Task1_P2.ipynb          Main training notebook
├── 📁 ag_news_bert_results/    Training outputs and logs
├── 📁 news_classifier/         Saved model and tokenizer
└── README.md                   This file
```

🎓 Learning Outcomes

This project demonstrates:

- **Transfer Learning**: Using pre-trained BERT for domain-specific tasks
- **Text Classification**: Multi-class classification techniques
- **Model Fine-tuning**: Advanced training strategies with Hugging Face
- **Performance Evaluation**: Comprehensive model assessment
- **Model Deployment**: Saving and loading trained models

🔄 Next Steps

Potential improvements and extensions:

- Implement cross-validation for robust evaluation
- Experiment with different BERT variants (RoBERTa, DistilBERT)
- Add model interpretability and attention visualization
- Deploy model as a REST API or web application
- Implement real-time news classification pipeline

📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset](https://paperswithcode.com/dataset/ag-news)

---

**Author**: Internship Project  
**Last Updated**: September 2025  
**Status**: Complete ✅
