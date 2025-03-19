# StudyTool: PDF Summarization and Question Generation

## Overview
StudyTool is a web-based application built with Streamlit that allows users to upload PDF documents, extract text, summarize content, and generate study questions using state-of-the-art NLP models. It is designed to assist students and researchers in quickly understanding key points from lengthy documents and creating self-assessment questions.

## Features
- Extract text from uploaded PDF documents.
- Summarize extracted text using advanced NLP models like **PEGASUS** and **T5**.
- Generate relevant study questions from the summarized text using **T5 Question Generation Model**.
- Customizable chunk size and batch processing for optimized performance.
- Debug mode for analyzing extracted text and chunk processing.
- Downloadable summaries and generated questions for offline use.

## Technologies Used
- **Python**
- **Streamlit** (for UI)
- **Hugging Face Transformers** (for NLP models)
- **PyPDF2** (for PDF text extraction)
- **NLTK** (for text processing)
- **RAKE** (for keyword extraction)
- **Pyttsx3** (for text-to-speech integration)
- **PdfAnnotator** (for annotation support)

## Installation
To run StudyTool locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/studytool.git
   cd studytool
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a PDF file.
2. Select a summarization model (PEGASUS Large, PEGASUS XSum, or T5 Small).
3. Adjust chunk size and batch settings if needed.
4. Click **Process PDF** to generate summaries and study questions.
5. Download the summary and questions as text files.


## Future Enhancements
- Add support for multiple languages.
- Improve question quality with GPT-based models.
- Enhance UI with interactive annotation features.
- Integrate cloud storage for saving summaries and questions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



