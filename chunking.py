import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf_with_metadata(pdf_path: str, chunk_size: int = 150, chunk_overlap: int = 30):
    """Reads a PDF, splits text recursively page-by-page, and binds chunks to metadata."""
    try:
        # 1. Open the document and grab global metadata
        doc = fitz.open(pdf_path)
        global_metadata = doc.metadata
        
        # 2. Configure the recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        all_processed_chunks = []

        # 3. Process the document page by page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            display_page_num = page_num + 1  # 1-indexed for human readability
            
            # Skip empty pages
            if not page_text.strip():
                continue
                
            # 4. Split the current page's text into smaller chunks
            page_chunks = text_splitter.split_text(page_text)
            
            # 5. Bind each chunk to its corresponding structural metadata
            for chunk in page_chunks:
                chunk_data = {
                    "text": chunk.strip(),
                    "metadata": {
                        "source_document": pdf_path,
                        "page_number": display_page_num,
                        "title": global_metadata.get("title", "Unknown"),
                        "author": global_metadata.get("author", "Unknown"),
                        "creation_date": global_metadata.get("creationDate", "Unknown")
                    }
                }
                all_processed_chunks.append(chunk_data)
                
        doc.close()
        return all_processed_chunks

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []


# --- Run and Test the Script ---
if __name__ == "__main__":
    # Replace with your actual PDF file path
    pdf_file =  r"C:\Users\om\OneDrive\Documents\spark_learning_guide_for_ml_systems (1).pdf"
    
    # Run the processing pipeline
    final_chunks = chunk_pdf_with_metadata(pdf_file, chunk_size=2000, chunk_overlap=100)
    
    # Display the structured output
    print("=" * 60)
    print(f"🚀 SUCCESS: Generated {len(final_chunks)} total metadata-bound chunks.")
    print("=" * 60)
    
    # Print the first 3 chunks as a sample
    for idx, item in enumerate(final_chunks[::]):
        print(f"\n🧩 CHUNK #{idx + 1}")
        print(f"📄 Page Reference: {item['metadata']['page_number']}")
        print(f"📜 Doc Title:     {item['metadata']['title']}")
        print(f"📝 Text Snippet:   \"{item['text']}\"")
        print("-" * 60)
