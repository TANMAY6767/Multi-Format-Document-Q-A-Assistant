# File Q&A (learning project)

Upload PDF, TXT, CSV, or Excel files, then ask questions about the loaded data. The app uses Docling for PDF layout and table extraction, LangChain for document chunks and vector search, Weaviate Cloud for storage, and OpenAI for embeddings and answers.

## Run on Windows

1. Add these values to `.env`:

   ```text
   OPENAI_API_KEY=your_openai_api_key
   WEAVIATE_URL=your_cluster_rest_endpoint
   WEAVIATE_API_KEY=your_weaviate_cloud_api_key
   ```

   In the Weaviate Cloud console, use the cluster's REST endpoint for `WEAVIATE_URL`.
2. Install requirements:

   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Run the app:

   ```powershell
   streamlit run app.py
   ```

## Limits to control API use

- Up to 5 files in one load; Streamlit blocks files over 5 MB and the batch limit is 15 MB.
- PDFs can have up to 10 pages. CSV and Excel sheets can have up to 10,000 rows; workbooks can have up to 5 sheets.
- A load can create up to 40,000 embedding tokens. The app checks this before calling OpenAI.
- Each chat allows 5 user messages, with up to 3 chats. Questions are limited to 2,000 characters and answers to 500 output tokens.
- Loading a new batch replaces the previous files and clears the chat histories so old answers do not refer to replaced data.

## How it works

1. **Load Data** reads each selected file. Docling extracts PDF layout and tables. Pandas reads CSV and Excel rows. TXT files are read as UTF-8.
2. Text is split into chunks. The app checks the total embedding token count before sending anything to OpenAI.
3. LangChain creates OpenAI embeddings and writes the chunks to a Weaviate Cloud collection.
4. **Chats** retrieves four relevant chunks for each message. The LLM receives those chunks and earlier messages in the selected chat, so it can use the conversation context.

The chats are kept in the current Streamlit session. The loaded collection persists in Weaviate Cloud. PDF table extraction and generated answers can be wrong; check important figures in the original file.
