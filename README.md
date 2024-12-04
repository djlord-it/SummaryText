#  **Text Summarizer**

Effortlessly generate concise summaries of lengthy text with this user-friendly summarization tool. Whether you're dealing with articles, reports, or any verbose document, the app leverages cutting-edge NLP techniques to deliver the key points in seconds.

---

## **Key Features**
- **Quick Summarization:** Paste any text and get an instant summary.
- **Smart Sentence Selection:** Extracts the most important sentences using proven NLP techniques.
- **Flexible Tokenization:** Supports both NLTK and SpaCy for processing sentences.
- **Error Resilience:** Includes fallback mechanisms to ensure functionality even if some libraries are unavailable.
- **Simple GUI:** powered by **PyQt5** for effortless use.

---

## **How It Works**

This summarizer uses a combination of classical NLP techniques and modern tools to break down, analyze, and summarize text efficiently.

### **1. Sentence Tokenization**
The first step is breaking the text into manageable chunks—sentences. 
- **Default Method:** Uses **NLTK** to identify sentence boundaries.
- **SpaCy (Optional):** For users who install SpaCy, the app leverages its **dependency parsing** and linguistic models to identify sentences more accurately, especially for complex text.

### **2. Stop Words Removal**
Commonly used words (e.g., "is," "the," "and") that don’t contribute to the meaning of the content are filtered out. This step improves the tool’s focus on meaningful terms.

### **3. Sentence Scoring with TF-IDF**
The app calculates a score for each word using **Term Frequency-Inverse Document Frequency (TF-IDF)**:
- **Term Frequency (TF):** Measures how often a word appears in a sentence.
- **Inverse Document Frequency (IDF):** Gives higher importance to rare words.
This ensures that critical terms receive higher weights, guiding the summarization process.

### **4. Comparing Sentences Using Cosine Similarity**
To identify relationships between sentences, the app computes **cosine similarity**:
- Sentences with similar content (shared important terms) are grouped together.
- This helps in identifying the most representative sentences for the summary.

### **5. Sentence Length Weighting (Optional)**
Longer sentences often contain more information. The app includes a **length weighting mechanism** to prioritize such sentences without biasing against shorter, equally important ones.

### **6. Generating the Summary**
The top-scoring sentences are selected, reordered, and combined to form a coherent summary. If the original text is short, the app intelligently decides to return the entire text instead.

### **7. Fallbacks and Resilience**
- **If SpaCy is unavailable:** The app seamlessly switches to NLTK for sentence tokenization.
- **If text preprocessing fails:** The tool returns the first few sentences to ensure no blank outputs.

---

## **Why SpaCy Integration?**
SpaCy adds an advanced layer to the tool:
- **Linguistic Precision:** SpaCy provides better handling of sentence boundaries, especially in complex or technical text.
- **Named Entity Recognition (NER):** Although not used in this project, SpaCy’s NER capabilities can enhance future features.
- **High Performance:** SpaCy is optimized for large-scale text processing, making it a valuable addition for users handling extensive documents.

**Don’t have SpaCy?🙂** No problem! The app works perfectly fine with NLTK, offering a reliable fallback.

---

## **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/SummaryTextAi.git
   cd SummaryTextAi
   #install the packages
   pip install -r requirements.txt
   #download the nltk Data
   python nltk_download.py
   #download the language model
   python -m spacy download en_core_web_sm
   #run
   python main.py
## Example

- input:
    "Climate change is one of the most pressing challenges facing the world today. It is caused by the increasing concentration of greenhouse gases in the atmosphere, mainly due to human activities such as deforestation, industrial processes, and the burning of fossil fuels. These activities trap heat in the Earth's atmosphere, causing global temperatures to rise. The impacts of climate change are already being felt, including rising sea levels, more frequent and severe heatwaves, and unpredictable weather patterns.
The consequences of climate change are particularly severe for vulnerable communities, including low-lying coastal areas and regions with limited resources to adapt. In addition to environmental consequences, climate change also poses significant social and economic risks. These include the displacement of populations, damage to infrastructure, and disruptions to agriculture and food production.
To address this crisis, world leaders and scientists are urging immediate action to reduce greenhouse gas emissions, increase the use of renewable energy, and protect ecosystems. Efforts are underway to transition to cleaner, more sustainable energy sources like solar, wind, and hydropower, as well as promote energy efficiency and conservation. Governments, businesses, and individuals all play a critical role in combating climate change and ensuring a sustainable future for generations to come."



- output:
    "The impacts of climate change are already being felt, including rising sea levels, more frequent and severe heatwaves, and unpredictable weather patterns. The consequences of climate change are particularly severe for vulnerable communities, including low-lying coastal areas and regions with limited resources to adapt governments, businesses, and individuals all play a critical role in combating climate change and ensuring a sustainable future for generations to come"