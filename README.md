# PDF Chat Application

Welcome to the PDF Chat Application! This application allows users to interact with PDF materials by uploading a PDF and chatting with the content. The text from the PDF is extracted, embedded, and stored in a vector database to enable efficient querying and interaction.

## Features

- **PDF Upload:** Easily upload PDF files for processing.
- **Text Extraction:** Extracts text content from the uploaded PDF.
- **Word Embedding:** Embeds the words and stores them in a vector database for efficient querying.
- **Interactive Chat:** Provides an intuitive interface for users to interact with the uploaded PDF content through chat.

## Project Structure

The project consists of the following key files:

- `app.py`: Contains Streamlit code to provide the user interface for interacting with the application.
- `functions.py`: Contains all necessary actions to extract text from the PDF, embed words, and store them in a vector database.
- `requirements.txt`: Lists all the libraries and dependencies required to run the application.

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.x
- Streamlit
- Other dependencies listed in `requirements.txt`

### Installation

1. **Clone the Repository:**

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF:**
   - Open the application in your browser.
   - Use the provided interface to upload a PDF file.

3. **Chat with the PDF Content:**
   - Once the PDF is uploaded and processed, start interacting with the content through the chat interface.

## License

This project is licensed under the MIT License.
