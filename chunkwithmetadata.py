import fitz  # PyMuPDF


def read_pdf_with_metadata(pdf_path: str):
    """Reads a PDF file and extracts text along with page numbers and document metadata."""
    try:
        # 1. Open the PDF document
        doc = fitz.open(pdf_path)

        # 2. Extract and print global document metadata
        metadata = doc.metadata
        print("=" * 60)
        print("📄 DOCUMENT METADATA")
      
        print(f"Title:         {metadata.get('title', 'N/A')}")
        print(f"Author:        {metadata.get('author', 'N/A')}")
        print(f"Subject:       {metadata.get('subject', 'N/A')}")
        print(f"Keywords:      {metadata.get('keywords', 'N/A')}")
        print(f"Creator:       {metadata.get('creator', 'N/A')}")
        print(f"Producer:      {metadata.get('producer', 'N/A')}")
        print(f"Creation Date: {metadata.get('creationDate', 'N/A')}")
        print(f"Mod Date:      {metadata.get('modDate', 'N/A')}")
        print(f"Total Pages:   {len(doc)}")
        print("=" * 60 + "\n")

        # 3. Iterate through each page to extract content
        print("📝 EXTRACTING PAGE CONTENT...")
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Human-readable page numbers start at 1
            display_page_num = page_num + 1

            # Extract plain text from the page
            page_text = page.get_text()

            print(f"--- [ Start of Page {display_page_num} ] ---")

            if page_text.strip():
                print(page_text.strip())
            else:
                print("[Visual Page or No Selectable Text Found]")

            print(f"--- [ End of Page {display_page_num} ] ---\n")

        # Close the document resource
        doc.close()

    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")


# --- Run the Script ---
if __name__ == "__main__":
    # Replace 'sample.pdf' with the path to your actual PDF file
    pdf_file_path =
    read_pdf_with_metadata(pdf_file_path)
