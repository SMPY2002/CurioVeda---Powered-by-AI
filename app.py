from gtts import gTTS
import tempfile
import base64
import uuid
import streamlit.components.v1 as components

import streamlit as st
import requests
from requests.exceptions import RequestException
import logging
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sklearn.preprocessing import normalize
import numpy as np
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
from boilerpy3 import extractors
import pytesseract
from PIL import Image
import io
from io import BytesIO
import re
import time
import plotly.express as px
import pandas as pd
import base64

# other python file import------------------------
from graph_ai import query_llm_for_chart_data
from graph_ai import query_llm_for_chart_summary
# ------------------------------------------------
# -----------
# Render chart dynamically
import plotly.express as px


load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Session state init
for key, default in {
    "urls": [], "scraped_data": {}, "vector_store": None, "chat_history": [],
    "greeting_shown": False, "visualization_candidate_data": None,
    "show_visualize_prompt": False, "popup_open": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


if "speak_triggered" not in st.session_state:
    st.session_state.speak_triggered = False

if "narration_playing" not in st.session_state:
    st.session_state.narration_playing = False

if "voice_lang" not in st.session_state:
    st.session_state.voice_lang = "English"

if "narration_state" not in st.session_state:
    st.session_state.narration_state = "stopped"  # Options: "playing", "paused", "stopped"
if "audio_base64" not in st.session_state:
    st.session_state.audio_base64 = ""



# Initialize session state for URLs and scraped data and Chat History
# if "urls" not in st.session_state:
#     st.session_state.urls = []

# if "scraped_data" not in st.session_state:
#     st.session_state.scraped_data = {}

# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "greeting_shown" not in st.session_state:
#     st.session_state.greeting_shown = False

# if "visualization_candidate_data" not in st.session_state:
#     st.session_state.visualization_candidate_data = None

# if "show_visualize_prompt" not in st.session_state:
#     st.session_state.show_visualize_prompt = False

# if "popup_open" not in st.session_state:
#     st.session_state.popup_open = False


# Greet the user if opening the app for the first time
if not st.session_state.greeting_shown:
    st.session_state.chat_history.insert(0, {
        "question": "System Greeting",
        "answer": "Hello! I'm CurioVeda, your AI assistant. How can I assist you today?"
    })
    st.session_state.greeting_shown = True


# Function to add URL
def add_url(url):
    if url and url not in st.session_state.urls:
        st.session_state.urls.append(url)
    elif not url:
        st.warning("URL cannot be empty.")
    elif url in st.session_state.urls:
        st.warning("URL already added.")

# Function to delete URL
def delete_url(idx):
    url_to_delete = st.session_state.urls[idx]
    st.session_state.urls.pop(idx)
    # Also remove scraped data associated with this URL
    if url_to_delete in st.session_state.scraped_data:
        del st.session_state.scraped_data[url_to_delete]

# Set up logging
logging.basicConfig(filename="app_logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Enhanced scrape_url with detailed error handling and logging
def scrape_url(url):
    try:
        start_time = time.time()  # Start the timer for scraping duration
        # Attempt to scrape content
        content = scrape_article(url)
        
        if not content or "error" in content:
            error_msg = f"Failed to scrape {url}. Reason: {content.get('error', 'Unknown error')}"
            logging.error(error_msg)
            return error_msg
        
        # Extract relevant content
        text_content = content['text']
        metadata = content['metadata']

        # Include metadata at the beginning of the content
        if metadata:
            metadata_content = f"Title: {metadata.get('title', 'No Title')}\n"
            metadata_content += f"Author: {metadata.get('author', 'No Author')}\n"
            metadata_content += f"Published Date: {metadata.get('date', 'No Date')}\n\n"
            text_content = metadata_content + text_content

        # Optionally, combine text content with image text or inline links if needed
        if content['image_texts']:
            text_content += "\n\n" + "\n".join(content['image_texts'])
        if content['inline_links']:
            text_content += "\n\nInline Links:\n" + "\n".join([f"{link['text']} ({link['url']})" for link in content['inline_links']])
        
        # Add headings and subheadings to enhance content structure
        text_content = enhance_with_headings(content['headings'], text_content)
        # Process the content
        text_content = text_content.strip()

        end_time = time.time()  # End the timer
        elapsed_time = round(end_time - start_time, 2)
        st.info(f"Scraping completed for {url} in {elapsed_time} seconds.")  # Display the timing

        if not text_content:
            logging.warning(f"No meaningful content found for {url}.")
            return f"No meaningful content found for {url}."
        
        return text_content
    
    except RequestException as e:
        logging.error(f"Network error while scraping {url}: {str(e)}")
        return f"Network error while scraping {url}. Please check the URL or your connection."
    except Exception as e:
        logging.error(f"Unexpected error while scraping {url}: {str(e)}")
        return f"Unexpected error while scraping {url}: {str(e)}"

# Function to scrape article with enhanced method
def scrape_article(url):
    try:
        # Step 1: Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Step 2: Extract main content with BoilerPy3 (Removes boilerplate content)
        extractor = extractors.ArticleExtractor()
        main_text = extractor.get_content(html_content)

        # Step 3: Parse full HTML with BeautifulSoup for further extraction
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract metadata (title, author, published date)
        metadata = {
            "title": soup.title.string if soup.title else "No title found",
            "meta_description": soup.find("meta", {"name": "description"})["content"]
            if soup.find("meta", {"name": "description"})
            else "No description found",
            "publication_date": soup.find("meta", {"name": "date"})["content"]
            if soup.find("meta", {"name": "date"})
            else "No publication date found",
        }

        # Step 4: Extract images and use OCR for text extraction
        image_texts = []
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url:
                try:
                    # Fetch and process the image
                    img_response = requests.get(img_url)
                    img = Image.open(BytesIO(img_response.content))
                    text = pytesseract.image_to_string(img, lang="eng").strip()
                    if text:
                        image_texts.append(text)
                except Exception:
                    continue

        # Step 5: Handle inline links, footnotes, and references
        inline_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            if href and text:
                inline_links.append({"text": text, "url": href})

        # Step 6: Extract headings to maintain document structure
        headings = extract_headings(soup)

        # Step 7: Clean and preprocess the extracted text
        # Remove excessive whitespace, special characters, and ads-related sections
        cleaned_text = re.sub(r"\s+", " ", main_text)  # Normalize whitespace
        cleaned_text = re.sub(r"Advertisement|Sponsored", "", cleaned_text, flags=re.IGNORECASE)

        # Return all content in a structured format
        return {
            "text": cleaned_text,
            "image_texts": image_texts,
            "inline_links": inline_links,
            "metadata": metadata,
            "headings": headings,
        }

    except Exception as e:
        return {"error": str(e)}

# Helper function to extract headings
def extract_headings(soup):
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append((tag.name, tag.get_text(strip=True)))
    return headings

# Function to add hierarchical structure (headings) to content
def enhance_with_headings(headings, text_content):
    # Prepend the headings to the content for better context clarity
    for level, heading in headings:
        text_content = f"Heading ({level}): {heading}\n" + text_content
    return text_content

# <------------------------------------------------------------------>

def detect_statistical_data(text):
    lines = text.strip().split("\n")
    rows = []
    for line in lines:
        parts = re.split(r'\t+|\s{2,}', line.strip())
        if len(parts) == 2 and re.match(r'^\d{4}$', parts[0]) and parts[1].replace(',', '').replace('.', '').isdigit():
            rows.append([parts[0], float(parts[1].replace(",", ""))])
        elif len(parts) >= 3 and re.match(r'^\d{4}$', parts[0]):
            try:
                val1 = float(parts[1].replace(",", ""))
                val2 = float(parts[2].replace(",", "").replace("%", "")) if parts[2] != '-' else None
                rows.append([parts[0], val1, val2])
            except: continue
    if rows:
        return pd.DataFrame(rows, columns=["Year", "Value"] if len(rows[0]) == 2 else ["Year", "Population (millions)", "Growth Rate (%)"])
    return None

def speak_text(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code, slow=False)
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.mp3")
    tts.save(file_path)

    # Base64 encode
    with open(file_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()

    # HTML5 Audio with JS toggle play/pause
    autoplay_html = f"""
    <script>
    var existing = document.getElementById("curioveda_audio");
    if (existing) {{
        existing.pause();
        existing.remove();
    }}

    var audio = document.createElement("audio");
    audio.id = "curioveda_audio";
    audio.src = "data:audio/mp3;base64,{b64_audio}";
    audio.autoplay = true;
    document.body.appendChild(audio);
    </script>
    """

    components.html(autoplay_html, height=0)
    os.remove(file_path)
    st.session_state.narration_playing = True


def prepare_audio(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code, slow=False)
    temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp3")
    tts.save(temp_file)
    with open(temp_file, "rb") as f:
        audio_bytes = f.read()
    st.session_state.audio_base64 = base64.b64encode(audio_bytes).decode()
    os.remove(temp_file)


def generate_audio_base64(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code, slow=False)
    path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp3")
    tts.save(path)
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    os.remove(path)
    return encoded


def stop_narration():
    stop_js = """
    <script>
    var audio = document.getElementById("curioveda_audio");
    if (audio) {
        audio.pause();
        audio.remove();
    }
    </script>
    """
    components.html(stop_js, height=0)
    st.session_state.narration_playing = False


# <----------------------------------------------------------------->
def extract_and_preprocess(documents):
    # Combine all documents into a single text
    text = " ".join([doc.page_content for doc in documents])

    # More robust cleaning (handling special characters and more)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    return text, documents[0].metadata  # Return text and metadata

# Semantic Chunking with Overlap
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def embed_and_postprocess(chunks, embeddings_model):
    embedded_chunks = embeddings_model.embed_documents(chunks)
    embedded_chunks_np = np.array(embedded_chunks)
    normalized_embeddings = normalize(embedded_chunks_np, axis=1)  # Normalize
    return list(zip(chunks, normalized_embeddings.tolist()))

def create_vector_store():
    if st.session_state.scraped_data:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize the embeddings model
        start_time = time.time()  # Start the timer
        # Convert scraped data (dict) into a list of Document objects
        documents = [
            Document(page_content=content, metadata={"url": url})
            for url, content in st.session_state.scraped_data.items()
        ]
        
        text, metadata = extract_and_preprocess(documents)
        chunks = chunk_text(text)
        text_embeddings = embed_and_postprocess(chunks, embeddings_model)  # Now includes (chunk, embedding) pairs

        # Generate embeddings and create FAISS vector store
        st.session_state.vector_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,  # Correct format
            embedding=embeddings_model
        )
        # Generate embeddings and create FAISS vector store
        # st.session_state.vector_store = FAISS.from_documents(splitted_docs, st.session_state.embeddings)
        end_time = time.time()  # End the timer
        elapsed_time = round(end_time - start_time, 2)
        st.success(f"Learning is successful in {elapsed_time} seconds!. Now you start asking anything related to provided content.")  # Display the timing
    else:
        st.warning("No scraped content available to Learn.")


def query_bot_with_context(question, context):
    """
    Handles user queries and retrieves context-aware responses from the AI model.
    Includes structured responses and gracefully handles edge cases.
    """
    # Check if vector store is initialized
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        st.error("Vector store is not initialized. Please scrape content and create the vector store first.")
        return {"text": "Vector store not initialized."}

    # Handle specific questions about the chatbot's identity
    if question.lower() in ["what is your name?", "who are you?", "what do i call you?"]:
        return {"text": "My name is CurioVeda. I'm here to assist you with your articles and queries!"}

    # Set up the retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

    # Define a prompt for professional and structured responses
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant skilled in generating structured, actionable, and professional responses. 
        Use the context and user query to generate the answer in the following format:
        1. Provide a detailed text response to address the user's query.
        2. If the response contains numerical data or lists, structure it in a table format (e.g., pandas DataFrame-like).
        3. Ensure that the numerical data is formatted and labeled clearly for visualization (e.g., column names like 'Year', 'Values').
        4. If somone ask about this type of question, like What is Your Name? then you can answer like "My name is CurioVeda. I'm here to assist you with your articles and queries!"
        **Guiding Principles**:
        - Always provide professional and well-structured text responses.
        - Include numerical data tables if relevant, formatted for easy visualization.
        - Avoid unnecessary data or verbose explanations.

        Use the context to generate the response:
        {context}

        Now, answer the user's question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Retrieve answer
    try:
        response = retrieval_chain.invoke({'context': context, 'input': question})
    except Exception as e:
        return {"text": f"An error occurred while generating the response: {str(e)}"}

    # Handle cases where the response is empty or irrelevant
    if not response or "answer" not in response or response["answer"].strip() == "":
        return {"text": "I'm here to assist with knowledge-based queries. Could you please ask a question related to the provided information?"}

    return {"text": response.get("answer", "I'm sorry, I couldn't generate a response.")}


# Set the page configuration FIRST
st.set_page_config(page_title="CurioVeda: Interactive Article Insights", page_icon="static/assistant.png", layout="wide")

# <----------------------------------------------------------------->
# Function to get base64 of an image
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Load custom CSS
with open("static/style.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

logo_path = "static/logo-curioveda.png"
hero_image_path = "static/Chatbot-background.webp" # Background image for the hero section
# <----------------------------------------------------------------->


# Header Section with Login/Signup Button
logo_base64 = get_image_base64(logo_path) if os.path.exists(logo_path) else ""
st.markdown(f"""
    <div class="header">
        <div class="header-logo">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
            <h1>CurioVeda</h1>
        </div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about-us">About Us</a>
            <a href="#contact-us">Contact Us</a>
            <a href="#login" class="login-btn">Login/Sign Up</a>
        </div>
        <div class="burger" onclick="toggleNav()">
            <div></div>
            <div></div>
            <div></div>
        </div>
    </div>
    <script>
        function toggleNav() {{
            var navLinks = document.getElementById("nav-links");
            navLinks.classList.toggle("active");
        }}
    </script>
""", unsafe_allow_html=True)

# Hero Section with Background Image
if os.path.exists(hero_image_path):
    hero_image_base64 = get_image_base64(hero_image_path)
    st.markdown(
        f"""
        <div class="section"style='
            background: url(data:image/png;base64,{hero_image_base64}) no-repeat center center;'>
            <h1>Get Insights from Articles with CurioVeda<br>At your fingertips</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error("Hero image not found!")

# Main Content Section
st.markdown("<div class='section-header'>Enter Your Article URLs and Ask Questions</div>", unsafe_allow_html=True)

# Left Section: URL Management
with st.container():
    left_section, col , right_section = st.columns([1.2,0.5,2.3])

    # Left Section
    with left_section:
        st.markdown('<h3 class="center-text">Enter the Article URL</h3>', unsafe_allow_html=True)

        # Text input box styled using the class "input-box"
        url_input = st.text_input("Enter URL:", placeholder="Paste URL here...", key="url_input", label_visibility="collapsed")

        # Add URL button
        if st.button("Add URL"):
            if len(st.session_state.urls) >= 10:
                st.warning("You can only add up to 10 URLs.")
            elif url_input.strip():
                add_url(url_input)
            else:
                st.warning("Please enter a valid URL.")

        # Display added URLs
        st.write("#### Added URLs:")
        for idx, url in enumerate(st.session_state.urls):
            # Wrap the URL display
            cols = st.columns([8, 1])
            formatted_url = "\n".join(url[i:i+40] for i in range(0, len(url), 40))  # Split URL into 80-character chunks
            cols[0].write(f"{idx + 1}. {formatted_url}")
            if cols[1].button("❌", key=f"delete_{idx}"):
                delete_url(idx)

        # Middle Section
        # Button to scrape content from all added URLs
        if st.button("Scrape All Content"):
            for url in st.session_state.urls:
                if url not in st.session_state.scraped_data:
                    with st.spinner("Scraping the website..."):
                        st.session_state.scraped_data[url] = scrape_url(url)
            st.success("Scraping completed!")
            # st.write("#### Scraped Data:")
            # for url, data in st.session_state.scraped_data.items():
            #     st.write(f"URL: {url}")
            #     st.write(data[:5000] + "..." if len(data) > 500 else data)

        if st.button("Start Learning"):
            with st.spinner("Learning from the Scrapped data..."):
                create_vector_store()
        else:
            st.warning("No vector embeddings available. Please scrape and process content first.")

        # Display scraped content
        st.write("#### Scraped Content:")
        if st.session_state.scraped_data:
            # for url, content in st.session_state.scraped_data.items():
            #     st.write(f"#### URL: {url}")
            #     st.write(content[:500] + "..." if len(content) > 500 else content)
            st.write("Scraped content is now available for further processing.")
        else:
            st.write("No content scraped yet.")

        
        if st.button("Clear All Data"):
            st.session_state.urls = []
            st.session_state.scraped_data = {}
            st.session_state.vector_store = None
            st.success("All data cleared.")
            st.rerun()

    # Right Section: Chatbot Interface
    with right_section:
        st.markdown('<h3 class="center-text">Chatbot</h3>', unsafe_allow_html=True)
        # Visualization mode toggle
        # st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        # st.session_state.visual_mode = st.toggle("🖼️ Enable CurioVeda Visualization Mode", value=False)
        # st.markdown("</div>", unsafe_allow_html=True)

        # === Voice Language Selection aligned with toggle ===
        lang_toggle_col, voice_lang_col = st.columns([7, 1.5])  # Adjust width to match your toggle size
        with lang_toggle_col:
            st.markdown("<div style='text-align: right; margin-top: 30px;'>", unsafe_allow_html=True)
            st.session_state.visual_mode = st.toggle("🖼️ Enable CurioVeda Visualization Mode", value=st.session_state.get("visual_mode", False), key="visual_mode_toggle")
            st.markdown("</div>", unsafe_allow_html=True)
        # Voice language selection
        with voice_lang_col:
            lang_map = {
                "English": "en",
                "Hindi (हिंदी)": "hi",
                "Gujarati (ગુજરાતી)": "gu",
                "Marathi (मराठी)": "mr",
                "Tamil (தமிழ்)": "ta"
            }

            selected_lang = st.selectbox(
                "🎙️ Voice Language",
                options=list(lang_map.keys()),
                index=list(lang_map.keys()).index(st.session_state.get("voice_lang", "English")),
                key="voice_lang_select",
            )

            st.session_state.voice_lang = selected_lang

        # === Modern Chat Input with Speaker Icon and Language Control ===
        # Chat input and speaker button in the same row
        chat_col, icon_col, btn_col2 = st.columns([10, 1, 1.5])  # Adjust width ratio as needed

        with chat_col:
            user_question = st.text_input(
                "Ask a question:",
                placeholder="Type your question here...",
                label_visibility="collapsed",
                key="chat_input"
            )

        # === Voice Playback Controls: Play/Pause/Resume and Conditional Stop ===
        with icon_col:
            icon_map = {
                "stopped": "▶️",
                "playing": "⏸️",
                "paused": "🔁"
            }
            tooltip_map = {
                "stopped": "Play narration",
                "playing": "Pause narration",
                "paused": "Resume narration"
            }

            icon = icon_map[st.session_state.narration_state]
            tooltip = tooltip_map[st.session_state.narration_state]

            if st.button(icon, key="voice_control", help=tooltip):
                if st.session_state.narration_state == "stopped":
                    if st.session_state.chat_history:
                        answer = st.session_state.chat_history[0]["answer"]
                        lang_code = lang_map[st.session_state.voice_lang]
                        st.session_state.audio_base64 = generate_audio_base64(answer, lang_code)
                        st.session_state.narration_state = "playing"
                elif st.session_state.narration_state == "playing":
                    st.session_state.narration_state = "paused"
                elif st.session_state.narration_state == "paused":
                    st.session_state.narration_state = "playing"

        with btn_col2:
            if st.session_state.narration_state in ["playing", "paused"]:
                if st.button("🛑 Stop", help="Click to stop the ongoing narration."):
                    st.session_state.audio_base64 = ""
                    st.session_state.narration_state = "stopped"


        if user_question:
            context = "\n".join([f"User: {c['question']}\nBot: {c['answer']}" for c in st.session_state.chat_history]) + f"\nUser: {user_question}"
            a = query_bot_with_context(user_question, context)
            # Prevent duplicate chat entries
            if not st.session_state.chat_history or st.session_state.chat_history[0].get("question") != user_question:
                st.session_state.chat_history.insert(0, {"question": user_question, "answer": a["text"]})

        if st.session_state.chat_history:
            qa_pair = st.session_state.chat_history[0]
            st.markdown(f"**CurioVeda:** {qa_pair['answer']}")

            # Speak response if icon clicked
            if st.session_state.speak_triggered:
                speak_text(qa_pair["answer"], lang_code=lang_map[st.session_state.voice_lang])
                st.session_state.speak_triggered = False  # reset

            if st.session_state.audio_base64:
                js_control = {
                    "playing": "audio.play();",
                    "paused": "audio.pause();",
                    "stopped": "audio.pause(); audio.currentTime = 0;"
                }[st.session_state.narration_state]

                audio_html = f"""
                <audio id="curio_audio" src="data:audio/mp3;base64,{st.session_state.audio_base64}" style="display:none"></audio>
                <script>
                var audio = document.getElementById("curio_audio");
                if (audio) {{
                    {js_control}
                }}
                </script>
                """
                components.html(audio_html, height=0)


            # Show visualization button if enabled and chartable data found
            if st.session_state.visual_mode:
                chart_result = query_llm_for_chart_data(qa_pair["answer"])
                if chart_result:
                    st.session_state.chart_data = chart_result
                    if st.button("📊 Visualize this Data", key="show_chart_button"):
                        st.session_state.show_viz_panel = True

            # Render Visualization Panel popup only if toggled ON and user clicked visualize
            if st.session_state.visual_mode and st.session_state.get("show_viz_panel", False):
                
                # Inject CSS styles once
                st.markdown("""
                    <style>
                        .viz-popup-box {
                            position: fixed;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            z-index: 999;
                            background-color: #ffffff;
                            max-width: 750px;
                            width: 90%;
                            padding: 25px;
                            border-radius: 12px;
                            border: 2px solid #c3d3e3;
                            box-shadow: 0 0 40px rgba(0, 0, 0, 0.2);
                            animation: fadeIn 0.4s ease-in-out;
                        }
                        .popup-container {
                            display: flex;
                            justify-content: center;
                        }
                    </style>
                """, unsafe_allow_html=True)

                # Use columns to center content
                with st.container():
                    # st.markdown("<div class='popup-container'><div class='viz-popup-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: red;'>CurioVeda Visualization</h3>", unsafe_allow_html=True)
                    # Close button in the top-right corner
                    close_btn_col = st.columns([0.9, 0.1])[1]
                    with close_btn_col:
                        if st.button("❌", key="close_popup"):
                            st.session_state.show_viz_panel = False
                            st.rerun()

                    chart_data = st.session_state.chart_data
                    df = chart_data["df"]
                    title = chart_data["title"]
                    recommended = chart_data["chart_type"]
                    chart_explanations = {
                        "bar": "Best for comparing quantities across categories.",
                        "line": "Ideal for trends over time.",
                        "pie": "Great for showing proportional shares.",
                        "histogram": "Used to show frequency distribution.",
                        "scatter3d": "Shows relationship among 3 continuous variables."
                    }
                    st.markdown(f"<h6 style='color: green;'>Recommended Chart Type: {recommended}</h6>", unsafe_allow_html=True)
                    st.markdown(f"🧠 **Why this chart?** {chart_explanations.get(recommended, '')}")
                    # st.markdown("### Select Chart Type")
                    # Chart type selection
                    chart_types = {
                        "📊 Bar": "bar",
                        "📈 Line": "line",
                        "🧩 Pie": "pie",
                        "📉 Histogram": "histogram",
                        "🌐 3D Scatter": "scatter3d"
                    }
                    # Default column selection for recommended chart
                    default_cols = list(df.columns[:3]) if len(df.columns) >= 3 else list(df.columns)

                    chart_placeholder = st.empty()

                    def render_recommended_chart():
                        fig = None
                        if recommended in ["bar", "line", "histogram"]:
                            if len(default_cols) >= 2:
                                x, y = default_cols[0], default_cols[1]
                                if recommended == "bar":
                                    fig = px.bar(df, x=x, y=y, title=title)
                                elif recommended == "line":
                                    fig = px.line(df, x=x, y=y, title=title)
                                elif recommended == "histogram":
                                    fig = px.histogram(df, x=y, title=title)
                        elif recommended == "pie":
                            if len(default_cols) >= 2:
                                fig = px.pie(df, names=default_cols[0], values=default_cols[1], title=title)
                        elif recommended == "scatter3d":
                            if len(default_cols) >= 3:
                                import plotly.graph_objects as go
                                fig = go.Figure(data=[go.Scatter3d(
                                    x=df[default_cols[0]],
                                    y=df[default_cols[1]],
                                    z=df[default_cols[2]],
                                    mode='markers',
                                    marker=dict(size=6, color=df[default_cols[2]], colorscale='Viridis', opacity=0.8)
                                )])
                                fig.update_layout(title=title)
                        if fig:
                            chart_placeholder.plotly_chart(fig, use_container_width=True, key="recommended_chart_main" if 'from_reset' not in st.session_state else "recommended_chart_reset")


                    render_recommended_chart()

                    # Allow user to update manually (expandable)
                    with st.expander("🔧 Column Selection for Custom Visualization"):
                        chart_label = st.selectbox("Choose Chart Type", list(chart_types.keys()), index=list(chart_types.values()).index(recommended))
                        chart_type = chart_types[chart_label]

                        selected_cols = st.multiselect("Select Columns to Visualize", options=list(df.columns), default=default_cols[:3])

                        if st.button("🔁 Reset to Recommended", key="reset_to_recommended"):
                            st.session_state.from_reset = True
                            render_recommended_chart()
                            st.session_state.from_reset = False
                        else:
                            fig = None
                            if chart_type in ["bar", "line", "histogram"]:
                                if len(selected_cols) >= 2:
                                    x, y = selected_cols[0], selected_cols[1]
                                    if chart_type == "bar":
                                        fig = px.bar(df, x=x, y=y, title=title)
                                    elif chart_type == "line":
                                        fig = px.line(df, x=x, y=y, title=title)
                                    elif chart_type == "histogram":
                                        fig = px.histogram(df, x=y, title=title)
                            elif chart_type == "pie":
                                if len(selected_cols) >= 2:
                                    fig = px.pie(df, names=selected_cols[0], values=selected_cols[1], title=title)
                            elif chart_type == "scatter3d":
                                if len(selected_cols) >= 3:
                                    import plotly.graph_objects as go
                                    fig = go.Figure(data=[go.Scatter3d(
                                        x=df[selected_cols[0]],
                                        y=df[selected_cols[1]],
                                        z=df[selected_cols[2]],
                                        mode='markers',
                                        marker=dict(size=6, color=df[selected_cols[2]], colorscale='Viridis', opacity=0.8)
                                    )])
                                    fig.update_layout(title=title)

                            if fig:
                                chart_placeholder.plotly_chart(fig, use_container_width=True, key="custom_chart")
                            else:
                                st.warning("Please select enough columns to render this chart type.")

                    st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("#### Chat History")
        for item in st.session_state.chat_history[1:]:
            st.markdown(f"**You:** {item['question']}")
            st.markdown(f"**CurioVeda:** {item['answer']}")

        if st.button("Clear Chat History"):
            st.session_state.chat_history.clear()
            st.success("Chat history cleared.")
            st.rerun()

# Footer Section
st.markdown("<div class='footer'>© 2024 QueryHub. All Rights Reserved.</div>", unsafe_allow_html=True)