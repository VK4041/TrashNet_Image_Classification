# GPT-Style Prompt Continuation

A deep learning project exploring transfer learning, fine-tuning, and Parameter-Efficient Fine-Tuning (PEFT) techniques for text generation using GPT-based models.

## 📋 Overview

This project demonstrates advanced natural language processing techniques by fine-tuning pretrained transformer models on specialized documents (UNDRIP and Economic Reports). It compares traditional full fine-tuning with modern PEFT approaches, specifically LoRA (Low-Rank Adaptation).

## 🎯 Key Features

- **Transfer Learning**: Leverages pretrained GPT-2 and GPT-Neo models
- **Document Processing**: Extracts and cleans text from PDF documents
- **Multiple Fine-Tuning Approaches**:
  - Full fine-tuning on domain-specific data
  - LoRA-based Parameter-Efficient Fine-Tuning
- **Comprehensive Evaluation**: Uses perplexity metrics and prompt continuation
- **Model Comparison**: Benchmarks baseline vs fine-tuned vs PEFT models

## 🏗️ Architecture

### Models Explored

1. **GPT-2** (124M parameters)
   - Base model for initial experiments
   - 12 transformer blocks with self-attention

2. **GPT-Neo-125M** (125M parameters)
   - Variant model showing improved generalization
   - Enhanced performance on unseen data

3. **LoRA PEFT Model**
   - Only 294,912 trainable parameters (0.24% of total)
   - 99.76% reduction in trainable parameters
   - Comparable performance to full fine-tuning

## 📊 Results Summary

| Model | UNDRIP PPL | Economic PPL | Trainable Params | Training Time |
|-------|------------|--------------|------------------|---------------|
| GPT-2 (Baseline) | 19.37 | 20.53 | 163M | N/A |
| GPT-2 (Fine-tuned) | 16.88 | 18.02 | 163M | ~800s |
| GPT-Neo (Baseline) | 19.37 | 20.53 | 125M | N/A |
| GPT-Neo (Fine-tuned) | 15.68 | 17.25 | 125M | ~2400s |
| LoRA (Fine-tuned) | 16.45 | 17.89 | 294K | ~1800s |

*Lower perplexity indicates better model performance*

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers peft accelerate pdfplumber torchinfo
```

### Installation

```bash
git clone https://github.com/VK4041/GPT-Style_Prompt_Continuation.git
cd GPT-Style_Prompt_Continuation
```

### Running the Notebook

1. Mount Google Drive (if using Colab)
2. Place your PDF documents in the appropriate directory
3. Run cells sequentially
4. Models will be saved to `./fine_tuned_gpt2`, `./fine_tuned_variant`, and `./fine_tuned_lora_peft`

## 📖 Project Structure

```
├── Data Extraction & Cleaning
│   └── PDF text extraction with preprocessing
├── Baseline Model Evaluation
│   └── Perplexity computation on pretrained models
├── Full Fine-Tuning
│   ├── GPT-2 fine-tuning
│   └── GPT-Neo fine-tuning
├── PEFT with LoRA
│   ├── LoRA configuration
│   ├── Efficient fine-tuning
│   └── Performance comparison
└── Evaluation & Analysis
    ├── Perplexity metrics
    └── Qualitative text generation
```

## 🔧 Training Configuration

### Hyperparameters

- **Learning Rate**: 1e-5 (full fine-tuning), 3e-5 (LoRA)
- **Epochs**: 3-5 with early stopping
- **Batch Size**: 2 (due to GPU constraints)
- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: Linear warmup (10% of steps)
- **Context Length**: 1024 tokens
- **Stride**: 512 tokens (for sliding window)

### Early Stopping

- **Patience**: 3 epochs
- **Min Delta**: 0.1 (full fine-tuning), 0.01 (LoRA)

## 💡 Key Techniques

### Data Processing

- Page-wise PDF text extraction using `pdfplumber`
- Whitespace normalization and header/footer removal
- Non-ASCII character filtering
- Sliding window chunking for long documents

### LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    bias="none"
)
```

### Perplexity Evaluation

Uses sliding window approach with overlapping chunks to compute model uncertainty on text sequences.

## 📈 Insights

### Pretrained Model
- General, bureaucratic tone
- Lacks domain-specific depth
- Broader generalization

### LoRA Fine-Tuned Model
- Data-driven, analytical approach
- Provides statistics with moderate interpretation
- **99.76% fewer parameters** than full fine-tuning
- Achieves near-equivalent performance

### Fully Fine-Tuned Model
- Strong domain adaptation
- Human-rights advocacy tone
- Best performance on unseen data
- Risk of slight overfitting

## 🎓 Academic Context

**Course**: SIT744 Deep Learning - Deakin University  
**Author**: Varun Kumar

This project demonstrates practical applications of:
- Transfer learning in NLP
- Domain adaptation techniques
- Parameter-efficient fine-tuning methods
- Model evaluation and comparison

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{kumar2024gpt_prompt_continuation,
  author = {Kumar, Varun},
  title = {GPT-Style Prompt Continuation with PEFT},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/VK4041/GPT-Style_Prompt_Continuation}
}
```

## ⚠️ Notes

- Designed for Google Colab with GPU support >= L4 High RAM GPU
- Requires significant computational resources
- PDF files included
- CUDA memory optimizations included for T4 GPU
- Trained on A100 GPU

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📧 Contact

Email me: varunvk4041@gmail.com

Find the resources here: https://drive.google.com/drive/folders/14TOWegOtAg8Tfdjy9NRyVN0jLoLlrlA7?usp=sharing

---

**Note**: This project was developed as part of academic coursework. The techniques demonstrated are applicable to various NLP tasks requiring domain adaptation with limited computational resources.
