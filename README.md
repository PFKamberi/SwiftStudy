# 🧠 Study Swift – Your Smart Study Companion

Study Swift is an intelligent multimodal study assistant designed to help students quickly extract, search, and understand key information from both video lectures and PDF study materials. By combining state-of-the-art speech transcription, document parsing, and semantic search technologies, Study Swift empowers learners to find relevant content instantly without sifting through entire files.

With Study Swift, students can:

  *  Upload video lectures in MP4 format and get accurate transcriptions automatically.

  * Upload PDF textbooks, slides, and notes and extract meaningful chunks of information.

  *  Perform natural language queries across all indexed materials, retrieving precise answers and relevant text snippets.

  *  Seamlessly switch between searching video content and text documents.

  *  Save time and boost comprehension by focusing on exactly what they need to know.

Under the hood, Study Swift leverages cutting-edge tools:

   * OpenAI Whisper for automatic speech recognition of videos.

   * Unstructured library for advanced PDF content parsing and chunking.

   * BAAI/bge-m3 ONNX model for high-quality semantic embeddings.

   * Qdrant vector database for lightning-fast similarity search.

   * Streamlit UI for easy, interactive access to all features.

Study Swift is ideal for students, educators, and lifelong learners who want to study more efficiently, retain information better, and unlock knowledge effortlessly.

## Setup Instructions

Before running the app, make sure you have the following prerequisites installed on your system:

  *  Python (version 3.8 or higher recommended)
    You can download Python from the official website: https://www.python.org/downloads/
    Verify installation by running python --version or python3 --version in your terminal.

  *  Rye package manager (used to manage dependencies and run the app)
    Install Rye by following the instructions at https://rye-up.com/

   * Docker (for running the Qdrant vector database locally)
    Download and install Docker from https://www.docker.com/get-started

## Running the Project

1. Sync dependencies using Rye:
    ```
    rye sync
    ```
2. Run Qdrant in a Docker container:
    ```
    docker run -d --name qdrant_makeathon -p 6333:6333 -v /path/to/qdrant/dir qdrant/qdrant
    ```

    * To stop Qdrant:
        ```
        docker stop qdrant_makeathon
        ```
    * To start Qdrant again:
        ```
        docker start qdrant_makeathon
        ```
3. Insert your OpenAI API key at the beggining of the streamlit_app.py
4. Run the Study Swift app using Streamlit:
    ```
    rye run streamlit run src/makeathon/streamlit_app.py
    ```
