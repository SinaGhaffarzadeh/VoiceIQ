
# VoiceIQ 🎙🧠

**VoiceIQ** is a voice-based assistant prototype that transforms spoken Persian (Farsi) language into text and injects it into a Language Model (LLM) for intelligent querying. The purpose of this project is to provide a natural, spoken interface for users to interact with powerful AI models using their voice — especially in Persian — making automation, Q&A, and task support more intuitive and hands-free.

---

## 🧠 What Is It?

This project is the **first step** toward building a voice-enabled assistant capable of:

- Listening to audio input through a microphone
- Converting speech to Persian text in real-time
- Sending that text to an LLM for analysis or question-answering
- Returning and displaying the LLM’s response to the user

You can think of VoiceIQ as a terminal-based “Voice ChatGPT” tailored for Persian speakers.

---

## 🎯 Features

- ✅ Real-time speech-to-text using `speech_recognition`
- ✅ Persian language reshaping and RTL text support
- ✅ HuggingFace-based embedding and language models
- ✅ GPU support (CUDA)
- ✅ Error handling for microphone and internet connection issues
- 🚧 Planned:
  - Auto-detection of intent (e.g., summarization, question answering)
  - GUI version using `QTPY`
  - Local, offline speech recognition
  - Retry mechanism when no speech is detected
  - Improved prompt injection and formatting

---

## 🛠 Tech Stack

### Speech-to-Text:
- `speechrecognition`
- `pyaudio`

### Persian Language Support:
- `arabic-reshaper`
- `python-bidi`

### LLM & Embedding:
- `llama-index`
- `transformers`
- `sentence-transformers`
- `huggingface_hub`
- `torch` with CUDA

---

## 🔧 Setup & Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/VoiceIQ.git
cd VoiceIQ
```

2. **Install dependencies:**

> Make sure you are using Python 3.12+ and have `pip` installed.

```bash
pip install -r requirements.txt
```

3. **Run the assistant:**

```bash
python voiceiq.py
```

4. **Authenticate with Hugging Face:**

Replace `login(token="...")` with your actual Hugging Face token.

---

## 📂 Folder Structure

```
VoiceIQ/
│
├── Data/                      # Folder for source documents (e.g., PDFs)
├── voiceiq.py                # Main application code
├── persian_language_converter.py  # Persian reshaping helper
├── README.md                 # You’re reading this!
└── requirements.txt          # List of required libraries
```

---


## 🙋‍♂️ Persian Language Helper Function

To display Persian text properly (right-to-left with reshaped characters), we use this utility function:

```python
def persian_lang_converter(text):
    import arabic_reshaper
    from bidi.algorithm import get_display
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
```

---

## 🔮 Future Goals

- Full offline speech-to-text (Whisper or Vosk integration)
- Smart command routing based on LLM classification
- Telegram or mobile-based interaction
- GUI (with QtPy) for enhanced usability

---

