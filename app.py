from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import pandas as pd
import pytesseract
from pdf2image import convert_from_bytes


# Page Config

st.set_page_config(page_title='Rocket Profit AI Resume Sorter', page_icon="🚀", layout="wide")
load_dotenv()

Secret_pass = os.getenv("APP_PASSWORD")

user_pass = st.sidebar.text_input("Enter Access Key", type='password')

if user_pass == Secret_pass:
    st.sidebar.success("Access Granted")
    

## Cached Models
@st.cache_resource

def init_models():
    embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

    return embeddings, llm

embeddings, llm = init_models()

## Extract Text

def extract_text(feed):
    text = ""
    try:
        # Step 1: Try standard extraction
        pdf_reader = PdfReader(feed)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        
        # Step 2: If no text was found, use OCR (The "Magic" Step)
        if not text.strip():
            # Reset feed pointer and convert PDF to images
            feed.seek(0)
            images = convert_from_bytes(feed.read())
            for image in images:
                text += pytesseract.image_to_string(image)
                
    except Exception as e:
        st.error(f"Error reading {feed.name}")
    
    return text

## UI Layout

st.title("🚀 Rocket Profit: Stop sorting resumes manually. Try this instead!")
st.markdown("### Revamping Operational Efficiency for HR Consulting")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_text = st.text_area("Paste the Job Description here...", height=250)
    
with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)
    
## Processing Section

if st.button("Start Sorting"):
    if not jd_text or not uploaded_files:
        st.warning("Please Provide both a Job Description and at least 1 Resume")
        
    else:
        with st.spinner("Analyzing and Ranking Candidates..."):
        
            all_extracted = [{"name": file.name, "text": extract_text(file)} for file in uploaded_files]
            
            
            resume_data = [r for r in all_extracted if r.get("text") and str(r["text"]).strip()]
            
            if len(resume_data) < len(all_extracted):
                st.warning(f"⚠️ {len(all_extracted) - len(resume_data)} file(s) were skipped because no readable text was found (possibly scanned images).")

            if not resume_data:
                st.error("No readable text found in any uploaded resumes. Process stopped.")
                st.stop()

            jd_vector = embeddings.embed_query(jd_text)
            resume_texts = [r["text"] for r in resume_data]
            resume_vectors = embeddings.embed_documents(resume_texts)

            scores = cosine_similarity([jd_vector], resume_vectors)[0]

            results = []
            for i, score in enumerate(scores):
                if score > 0.3:
                    prompt = f""" You are an expert HR consultant. Critically evaluate this resume against the JD. 
                    Resume: {resume_texts[i]} 
                    JD: {jd_text}
                    
                    Provide a highly structured response:
                    - 3 Pros: (Bullet points)
                    - 1 Critical Con/Missing Skill:
                    - Overall Verdict: (2-4 line summary)
                    """
                    analysis = llm.invoke(prompt)
                    results.append({
                        "Name" : resume_data[i]["name"],
                        "Similarity" : round(score * 100, 1),
                        "AI Analysis" : analysis.content
                        })
            ## Sort by Highest Score
            results = sorted(results, key=lambda x: x["Similarity"], reverse = True )

            st.success(f"✅ Successfully analyzed {len(uploaded_files)} resumes!")

            for res in results:
                with st.expander(f"{res['Name']} - Semantic Match: {res['Similarity']}%"):
                    st.markdown(res["AI Analysis"])
                    
            if results:
                df = pd.DataFrame(results)
                # Remove the long analysis for a clean CSV
                csv = df[['Name', 'Similarity']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Ranking Report (CSV)",
                    data=csv,
                    file_name='resume_sorting.csv',
                    mime='text/csv',
                    )

else:
    st.sidebar.error("Acess Denied")
    st.stop()
    