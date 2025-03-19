from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st
import PyPDF2
import nltk
import traceback
import time
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
import pyttsx3
from pdf_annotate import PdfAnnotator, Location, Appearance

# Load tokenizer and model for summarization with caching
@st.cache_resource
def load_summarizer_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None, None

# Load question generation pipeline with caching
@st.cache_resource
def load_question_generator():
    try:
        return pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
    except Exception as e:
        st.error(f"Error loading question generation model: {str(e)}")
        return None

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Split text into smaller chunks with improved size
def split_text_into_chunks(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Function to summarize text with a sliding window approach
def summarize_text_with_sliding_window(text, tokenizer, model, timeout=30):
    if not text or len(text.strip()) < 50:
        return ""
    
    start_time = time.time()
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Check for timeout
        if time.time() - start_time > timeout:
            st.warning("Summarization timed out")
            return text[:200] + "..."
            
        summary = model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=50,
            max_length=200,
            length_penalty=2.0,
            early_stopping=True
        )
        
        result = tokenizer.decode(summary[0], skip_special_tokens=True)
        
        # Verify quality - simple check for meaningful content
        if len(result) < 20 or result == text[:len(result)]:
            return text[:200] + "..."
            
        return result
    except Exception as e:
        st.warning(f"Summarization error: {str(e)}")
        return text[:200] + "..."

# Function to generate questions with error handling
def generate_questions(text, question_generator, timeout=20):
    if not question_generator or not text or len(text.strip()) < 50:
        return []
        
    start_time = time.time()
    try:
        questions = question_generator(
            text, 
            max_length=512, 
            num_return_sequences=3, 
            num_beams=4
        )
        
        # Filter out poor quality questions
        filtered_questions = []
        for q in questions:
            question_text = q['generated_text']
            # Basic quality filtering
            if (len(question_text) > 10 and
                '?' in question_text and
                not question_text.startswith("What is the ") and
                question_text not in filtered_questions):
                filtered_questions.append(q)
                
        return filtered_questions
    except Exception as e:
        st.warning(f"Question generation error: {str(e)}")
        return []

# Streamlit app logic
def main():
    st.set_page_config(page_title="StudyTool", layout="wide")
    
    st.title("StudyTool")
    st.write("Upload a PDF to generate summaries and study questions")
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Model selection for summarization
    summarizer_options = {
        "PEGASUS Large (Better quality, slower)": "google/pegasus-large",
        "PEGASUS XSum (Faster, more concise)": "google/pegasus-xsum",
        "T5 Small (Fastest)": "t5-small"
    }
    
    selected_summarizer = st.sidebar.selectbox(
        "Select Summarization Model",
        options=list(summarizer_options.keys()),
        index=0
    )
    
    # Configure chunk and batch sizes
    chunk_size = st.sidebar.slider("Chunk Size (characters)", 500, 2000, 1000)
    batch_size = st.sidebar.slider("Batch Size", 1, 10, 3)
    
    # Upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    if not pdf_file:
        st.info("Please upload a PDF file to begin")
        return
        
    # Extract text from the uploaded PDF
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(pdf_file)
    
    # Display the extracted text for verification if in debug mode
    if debug_mode:
        st.subheader("Extracted PDF text:")
        st.text_area("PDF Content", pdf_text, height=200)

    if not pdf_text:
        st.error("Could not extract text from the PDF. Please try another file.")
        return
        
    # Load models based on selection with a loading indicator
    with st.spinner("Loading AI models (this may take a moment)..."):
        model_name = summarizer_options[selected_summarizer]
        
        # Override the cache function temporarily to use the selected model
        @st.cache_resource
        def load_selected_summarizer():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                return tokenizer, model
            except Exception as e:
                st.error(f"Error loading summarization model: {str(e)}")
                return None, None
                
        summarizer_tokenizer, summarizer_model = load_selected_summarizer()
        question_generator = load_question_generator()
    
    if not summarizer_tokenizer or not summarizer_model:
        st.error("Failed to load summarization model. Please try refreshing the page.")
        return
        
    if not question_generator:
        st.warning("Failed to load question generator model. Continuing with summarization only.")

    # Process button
    if st.button("Process PDF"):
        # Split text into smaller chunks
        with st.spinner("Splitting text into chunks..."):
            text_chunks = split_text_into_chunks(pdf_text, max_chunk_size=chunk_size)
            
        if debug_mode:
            st.write(f"Split into {len(text_chunks)} chunks")

        # Initialize empty strings for summary and questions
        full_summary = ""
        all_questions = []

        # Create columns for summary and questions
        col1, col2 = st.columns(2)
        
        with col1:
            summary_container = st.empty()
            st.subheader("Summary")
            summary_placeholder = st.empty()
            
        with col2:
            questions_container = st.empty()
            st.subheader("Generated Questions")
            questions_placeholder = st.empty()

        # Add a progress bar for user feedback
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process each chunk in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_summary = ""
            batch_questions = []
            
            status_text.text(f"Processing chunks {i+1} to {min(i+batch_size, len(text_chunks))} out of {len(text_chunks)}")
            
            try:
                for j, chunk in enumerate(batch):
                    if len(chunk) > 50:  # Only process meaningful chunks
                        if debug_mode:
                            st.write(f"Processing chunk {i+j+1}: {chunk[:50]}...")
                            
                        # Generate summary for this chunk
                        chunk_summary = summarize_text_with_sliding_window(
                            chunk, summarizer_tokenizer, summarizer_model
                        )
                        
                        if chunk_summary:
                            batch_summary += chunk_summary + "\n\n"
                            
                            # Generate questions from the summary
                            if question_generator:
                                chunk_questions = generate_questions(chunk_summary, question_generator)
                                batch_questions.extend(chunk_questions)
                
                full_summary += batch_summary
                all_questions.extend(batch_questions)
                
                # Update display in real-time
                summary_placeholder.markdown(full_summary)
                
                if all_questions:
                    questions_text = "\n\n".join([f"{idx + 1}: {q['generated_text']}" 
                                            for idx, q in enumerate(all_questions)])
                    questions_placeholder.markdown(questions_text)
                
            except Exception as e:
                st.error(f"Error processing batch {i}: {str(e)}")
                if debug_mode:
                    st.code(traceback.format_exc())

            # Update the progress bar
            progress = min((i + batch_size) / len(text_chunks), 1.0)
            progress_bar.progress(progress)

        # Final display update
        status_text.text("Processing complete!")
        
        # Check if we actually generated a summary and questions
        if not full_summary.strip():
            summary_placeholder.error("Failed to generate a summary. The model may not be working properly with this content.")
        
        if not all_questions:
            questions_placeholder.warning("No questions were generated. Try adjusting the settings or using a different PDF.")
        
        # Add download buttons
        if full_summary.strip():
            st.download_button(
                label="Download Summary",
                data=full_summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            
        if all_questions:
            questions_text = "\n\n".join([f"{idx + 1}: {q['generated_text']}" 
                                    for idx, q in enumerate(all_questions)])
            st.download_button(
                label="Download Questions",
                data=questions_text,
                file_name="study_questions.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()