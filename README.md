# <center>Local LLM chatbot</center>

## Models used:

> TinyLlama1.1b ==> 5 min to respond

> MBZUAI/LaMini-Cerebras-1.3B ==> 5 min

> MBZUAI/LaMini-GPT-1.5B ==> 2.5 min on average

- finalised third model due to performance and speed of response.

## Test results

![Alt text](./Local_LLM_bot.png)

- Note
  - The chatbot is enable with memory using sqlite3 database with django.

## Libraries used

- django
- python-dotenv
- langchain
- PyPDF2
- faiss-cpu
- sentence_transformers
- InstructorEmbedding
- chromadb
- tiktoken
- torch
- accelerate
- txtai
