from django.shortcuts import render
from django.http import JsonResponse
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline

import torch

device = torch.device('cpu')
# checkpoint = "MBZUAI/LaMini-T5-738M"
checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=512)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 1024,
        do_sample = True,
        temperature = 0.01,
    )
model = HuggingFacePipeline(pipeline=pipe)


# Set up prompts for standalone question, retriever, and answer templates
standaloneQuestionTemplate = "Given a question, convert it to a standalone question. question: {question} standalone question: "
standaloneQuestionPrompt = PromptTemplate.from_template(standaloneQuestionTemplate)


persist_directory = './db'
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # create the open-source embedding function
vectordb = Chroma(persist_directory=persist_directory, # Now we can load the persisted database from disk, and use it as normal.  
                  embedding_function=embeddings)
retriever = vectordb.as_retriever()

def combine_documents(docs):
    docs = '\n'.join(doc.page_content for doc in docs)
    return docs

def retrieve_standalone_question(resp):
    return resp.get("standalone_question")


answerTemplate = '''You are a helpful and enthusiastic support bot who can answer a given question 
based on the context provided. Try to find the answer in the context. 
Encourage a casual conversation if the used tells about his name or ask about yourself, 
like a friendly conversation.
But the priority is to find the answer form the context. If you really don't know the answer, say 
"I'm sorry, I don't know the answer to that." and direct the questioner to email help@company.com f
or human assitance. Don't try to make up an answer. Always speak as if you were chatting to a friend. 
Keep the conversations short to medium.
context: {context}
question: {question}
answer:
'''
answerPrompt = PromptTemplate.from_template(answerTemplate)

# CHAINS to combine the above created prompts
standaloneChain = standaloneQuestionPrompt | model | StrOutputParser()
retrieverChain = RunnableLambda(retrieve_standalone_question) | retriever | RunnableLambda(combine_documents)
answerChain = answerPrompt | model | StrOutputParser()

# chain = ({
#     "standalone_question": standaloneChain,
#     "original_input": RunnablePassthrough()
# } | 
# {
#     "context": retrieverChain,
#     "question": lambda original_input: original_input.question,
# } 
# | answerChain
# )

# message = 'Who are you, and what do you do?, and who am I?'
# response = chain.invoke({'question': message})
# print(response)

def ai_response(message):
    # Do something with the message here using LLM
    context = ({
        "standalone_question": standaloneChain,
        "original_input": RunnablePassthrough()
    } | retrieverChain )
    retriever_response = context.invoke({'question': message})
    ai_message = answerChain.invoke({'context': retriever_response[:512], 'question': message})
    return ai_message

# Create your views here.
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        # Do something with the message here using LLM
        ai_message = ai_response(message)

        return JsonResponse({'message': message, 'response': ai_message})
    return render(request, 'chatbot.html')