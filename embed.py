import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


# Set your Free Hugging Face API key here
HF_API_KEY = "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"


def chunk_and_embed_via_api_with_batching(
    pdf_path: str, 
    chunk_size: int = 150, 
    chunk_overlap: int = 30, 
    batch_size: int = 16  # Send 16 chunks per API request
):
    """Extracts text from PDF and sends chunks to the free Hugging Face API using batching."""
    try:
        # 1. Extract text and compile chunks (From our previous step)
        doc = fitz.open(pdf_path)
        global_metadata = doc.metadata
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        all_processed_chunks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if not page_text.strip():
                continue
                
            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                chunk_data = {
                    "text": chunk.strip(),
                    "metadata": {
                        "source_document": pdf_path,
                        "page_number": page_num + 1,
                        "title": global_metadata.get("title", "Unknown")
                    }
                }
                all_processed_chunks.append(chunk_data)
        doc.close()

        total_chunks = len(all_processed_chunks)
        if total_chunks == 0:
            print("No text chunks found to embed.")
            return []

        # 2. Initialize the Remote Inference API Embedding Engine
        print("🌐 Connecting to Hugging Face Inference API...")
        embedding_engine = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_API_KEY, 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 3. Batching Logic
        print(f"📦 Found {total_chunks} chunks. Processing in batches of {batch_size}...")
        all_embeddings = []

        for i in range(0, total_chunks, batch_size):
            # Extract the current batch of text chunks
            current_batch = all_processed_chunks[i : i + batch_size]
            batch_texts = [item["text"] for item in current_batch]
            
            print(f"  -> Sending batch { (i // batch_size) + 1 } (Chunks {i} to {i + len(batch_texts) - 1})...")
            
            # Request embeddings for the current batch
            batch_embeddings = embedding_engine.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Optional: Short pause to be polite to the free API server and avoid rate limits
            time.sleep(0.5)

        # 4. Pack vectors back into our structured objects
        for idx, vector in enumerate(all_embeddings):
            all_processed_chunks[idx]["embedding"] = vector

        return all_processed_chunks

    except Exception as e:
        print(f"Error during API chunking and embedding: {e}")
        return []


# --- Run and Test the Script ---
if __name__ == "__main__":
    pdf_file = r"C:\Users\om\OneDrive\Documents\spark_learning_guide_for_ml_systems (1).pdf"  # Replace with your actual PDF file path
    
    # Process the PDF using a batch size of 16
    embedded_data = chunk_and_embed_via_api_with_batching(
        pdf_file, 
        chunk_size=200, 
        chunk_overlap=30, 
        batch_size=16
    )
    
    if embedded_data:
        print("=" * 60)
        print("✅ BATCHED CLOUD PIPELINE COMPLETE")
        print(f"📊 Total Chunks Successfully Processed: {len(embedded_data)}")
        print("=" * 60)
