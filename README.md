# Telugu NER & Number-to-Words Conversion Pipeline

A comprehensive Python-based application for processing Telugu, Hindi, and English text to convert numeric expressions into their word equivalents using Named Entity Recognition (NER) and Large Language Models.

## 🎯 Features

- **Multilingual Support**: Telugu, Hindi, and English text processing
- **Advanced NER**: Uses AI4Bharat's IndicNER model for accurate entity detection
- **Number Conversion**: Converts all numeric expressions (digits, ordinals, decimals) to words
- **Language Detection**: Automatic language validation using langdetect
- **GPU Acceleration**: CUDA support for faster processing
- **LLM-Based Processing**: Google Gemma 3 for intelligent text transformation

## 📋 Project Structure

```
Telugu-NER/
├── README.md                    # This file
├── telugu_new_code.py          # Telugu numbers-to-words converter
├── telugu_ner_code.py          # Telugu NER + text normalizer
├── english_new_code.py         # English numbers-to-words converter
├── hindi_new_code.py           # Hindi numbers-to-words converter
├── telugu_new_code             # Telugu processing module
└── .gitignore                  # Git ignore configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (optional, for faster processing)
- CUDA toolkit (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sujithsandilya/Telugu-NER.git
cd Telugu-NER
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install transformers torch langdetect sentencepiece accelerate
```

## 📖 Usage Guide

### Telugu Text Processing

**Convert Telugu numbers to words:**
```bash
python telugu_new_code.py input.txt output.txt
```

**Example:**
- **Input**: "అతను 5వ మార్చి 2023న ఉదయం 11 గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు"
- **Output**: "అతను ఐదవ మార్చి రెండు వేల ఇరవై మూడు సంవత్సరంలో ఉదయం పదకొండు గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు"

### Telugu NER + Text Normalization

**Process Telugu text with NER:**
```bash
python telugu_ner_code.py input.txt output.txt
```

Default usage (auto-generates output file):
```bash
python telugu_ner_code.py input.txt
```

### English Text Processing

**Convert English numbers to words:**
```bash
python english_new_code.py input.txt output.txt
```

**Example:**
- **Input**: "he went to the school on 5th March 2023 at 11am through bus"
- **Output**: "he went to the school on fifth March two thousand and twenty three at eleven AM through bus"

### Hindi Text Processing

**Convert Hindi numbers to words:**
```bash
python hindi_new_code.py input.txt output.txt
```

## 🔧 Configuration

### Model Settings

You can customize the models used in the pipeline by editing the configuration section in each script:

```python
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NER_MODEL_NAME = "ai4bharat/IndicNER"
LLM_MODEL_NAME = "google/gemma-3-1b-it"    # Change to gemma-3-4b-it for better quality
DEVICE = 0 if torch.cuda.is_available() else -1  # Use GPU if available
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
```

### Model Options

- **NER Model**: `ai4bharat/IndicNER` - Multilingual entity recognition
- **LLM Models**: 
  - `google/gemma-3-1b-it` (faster, lightweight)
  - `google/gemma-3-4b-it` (slower, better quality)

## 🏗️ Processing Pipeline

### General Pipeline Flow

1. **Language Detection**: Validate input language using langdetect
2. **NER Processing**: Extract named entities using AI4Bharat/IndicNER model
3. **Number Classification**: Identify numeric expressions (cardinal, ordinal, decimal)
4. **LLM Transformation**: Use Gemma 3 to convert numbers to words
5. **Output Generation**: Write normalized text to output file

### Supported Number Types

- **Cardinal Numbers**: 5 → "five" (English), "ఐదు" (Telugu)
- **Ordinal Numbers**: 5th → "fifth" (English), "ఐదవ" (Telugu)
- **Decimal Numbers**: 5.5 → "five point five" (English)
- **Date Formats**: 2023-03-05 → "two thousand twenty-three March fifth"
- **Time Formats**: 11:30 → "eleven thirty"

## 📊 Performance Considerations

### GPU vs CPU

- **GPU (CUDA)**: ~2-5x faster for large batches
- **CPU**: Slower but works on any machine

### Model Size Impact

- **1B Model** (gemma-3-1b-it): Faster, uses less memory
- **4B Model** (gemma-3-4b-it): Slower, better quality output

### Recommended Setup

For optimal performance:
- Use GPU with CUDA support
- Use 4B model for production quality
- Process text in batches for efficiency

## 🐛 Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers --upgrade
```

### "CUDA out of memory"
- Use CPU instead: Set `DEVICE = -1`
- Use smaller model: Change to `gemma-3-1b-it`
- Process smaller text chunks

### "langdetect" errors
```bash
pip install langdetect
```

### Model download issues
- Check internet connection
- Ensure sufficient disk space (~5-10GB)
- Models auto-download from Hugging Face on first run

## 📝 Input/Output Format

### Input File Format

Plain text file (.txt) with content in the target language:
```
అతను 5వ మార్చి 2023న ఉదయం 11 గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు
```

### Output File Format

Plain text file (.txt) with numbers converted to words:
```
అతను ఐదవ మార్చి రెండు వేల ఇరవై మూడు సంవత్సరంలో ఉదయం పదకొండు గంటలకు బస్సులో పాఠశాలకు వెళ్ళాడు
```

## 🔐 Requirements

### Core Dependencies
- `transformers` - HuggingFace model loading
- `torch` - Deep learning framework
- `langdetect` - Language detection
- `sentencepiece` - Tokenization
- `accelerate` - Hardware acceleration

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 10-15GB for model downloads
- **GPU** (optional): NVIDIA GPU with 2GB+ VRAM

## 📚 Model Information

### AI4Bharat/IndicNER
- Multilingual NER model
- Supports 11 Indian languages including Telugu, Hindi, English
- Entity types: Person, Location, Organization, etc.

### Google Gemma 3
- Open-source language model
- Available in 1B and 4B variants
- Efficient for number conversion tasks

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests to improve the pipeline.

## 📄 License

This project is open source and available for educational and research purposes.

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository:
https://github.com/sujithsandilya/Telugu-NER/issues

## 🙏 Acknowledgments

- AI4Bharat for the IndicNER model
- Google for Gemma 3 LLM
- HuggingFace for model hosting and transformers library
