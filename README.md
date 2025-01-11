# CurioVeda - RAG-Based Chatbot for Efficient Analysis of Articles, News, Reports

## Overview

This project is my **final-year major project** that implements a chatbot capable of efficiently answering user queries based on articles, reports, and news scraped from user-provided URLs. The chatbot offers **multilingual support (Currently working on it)** and provides **graphical analysis(Currently working on it)** of tabular data to facilitate better insights.

The primary objective of this project is to significantly improve article analysis by extracting critical insights quickly and accurately. The project demonstrates advanced capabilities in web scraping, natural language processing (NLP), and data visualization.

---

## Key Features

1. **Efficient Web Scraping**: 
   - Scrapes content from user-provided URLs, including JavaScript-heavy websites, using Selenium.
   - Extracts text from images in articles using OCR with Tesseract.

2. **Data Preprocessing**:
   - Basic preprocessing of scraped data for optimal analysis.
   - Recursive text splitting with LangChain for manageable chunk sizes.

3. **Generative AI for Querying**:
   - Uses Google Generative AI embeddings to convert text into fixed-length vectors.
   - Stores processed content in a **FAISS Vector Store** for efficient similarity searches.
   - Employs LLaMA text generation model for accurate and context-relevant answers based on similarity search results.

4. **Multilingual Support**:
   - Provides responses in multiple languages, enabling accessibility for diverse users.

5. **Graphical Analysis**:
   - Analyzes tabular data and generates graphical visualizations to present insights in an intuitive format.

---

## Thought Process Behind the Project

This project was designed with the following considerations:

1. Enable efficient and comprehensive data extraction from user-provided URLs, including:
   - Dynamic content (JavaScript-heavy websites).
   - Embedded image-based text using OCR techniques.

2. Enhance the quality of responses by applying advanced preprocessing techniques and leveraging LangChain for effective text chunking.

3. Utilize powerful AI models (e.g., Google Generative AI embeddings and LLaMA) for robust similarity search and context-aware responses.

4. Empower users with multilingual interactions and graphical insights for tabular data, making the chatbot a versatile tool for analysis.

---

## Tech Stack

### **Languages and Frameworks**:
- Python
- Streamlit (for hosting and front-end interface)

### **Libraries**:
- **Web Scraping**: Selenium, BeautifulSoup
- **OCR**: Tesseract
- **NLP**: LangChain, LLaMA, FAISS Vector Store, Google Generative AI
- **Data Visualization**: Matplotlib, Seaborn, Plotly

### **Tools**:
- GitHub (Version Control)
- Streamlit Cloud (Hosting)

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SMPY2002/CurioVeda---Powered-by-AI.git
   cd CurioVeda---Powered-by-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   **Preview**
   ![Screenshot 2025-01-11 102503](https://github.com/user-attachments/assets/51b52f5d-b3cf-4329-86dc-9cfd2b52748c)
   ![Screenshot 2025-01-11 102527](https://github.com/user-attachments/assets/3ace5f5e-9f45-47fe-9d47-21e255e2b05b)


5. **Usage**:
   - Input the URLs containing articles/reports/news.
   - Query the chatbot in your preferred language.
   - View graphical insights for any tabular data provided.

---

## Future Scope

- Add support for more advanced AI models and embedding techniques.
- Expand multilingual capabilities to include more languages.
- Integrate real-time streaming data analysis.
- Enhance graphical analysis features to include predictive insights.
- Optimize the backend for faster query responses and lower resource usage.

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this project as per the license terms.

---

## Contact
For any queries or suggestions, please reach out via:
- Email: <smpy1405@gmail.com>
- LinkedIn: [Shivam Pandey](https://www.linkedin.com/in/shivam-pandey1405)
