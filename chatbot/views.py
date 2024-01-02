from django.shortcuts import render
from django.http import JsonResponse
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from txtai.pipeline import LLM
from langchain.llms import Anthropic 
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline, AutoModelForCausalLM
import asyncio
import torch

device = torch.device('cpu')
checkpoint = "MBZUAI/LaMini-GPT-1.5B"
# checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# checkpoint = "MBZUAI/LaMini-Cerebras-1.3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=512)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map=device,
#     torch_dtype=torch.float32
# )
base_model = AutoModelForCausalLM.from_pretrained(checkpoint)

pipe = pipeline(
        'text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 1024,
        do_sample = True,
        temperature = 0.3,
    )
model = HuggingFacePipeline(pipeline=pipe)
# model = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


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


answerTemplate = '''
> You are a helpful and enthusiastic support bot who can answer a given question 
based on the context provided. Along with your knowledge use the context provided and user previous questions to give answer to the question. 

> If you really don't know the answer, say "I'm sorry, I don't know the answer to that." 
and then direct the questioner to contact help@company.com for human assitance. 

> Always speak as if you were chatting to a friend. 

context: {context}

previous_questions: {previous_questions}

question: {question}

answer:
'''
answerPrompt = PromptTemplate.from_template(answerTemplate)

# CHAINS to combine the above created prompts
standaloneChain = standaloneQuestionPrompt | model | StrOutputParser()
retrieverChain = RunnableLambda(retrieve_standalone_question) | retriever | RunnableLambda(combine_documents)
answerChain = answerPrompt | model | StrOutputParser()


async def util1(message):
    context = ({
        "standalone_question": standaloneChain,
        "original_input": RunnablePassthrough()
    } | retrieverChain )
    retriever_response = context.invoke({'question': message})
    return retriever_response
    

async def ai_response(message):
    # Do something with the message here using LLM
    retriever_response = await util1(message)
    if len(retriever_response) > 512:
        retriever_response = retriever_response[:512]
    ai_message =  answerChain.invoke({'context': retriever_response, 
                                      'question': message,
                                      'previous_questions': previous_questions})
    return ai_message

global previous_questions
previous_questions = []

# Create your views here.
async def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        previous_questions.append(message)
        if len(previous_questions) > 10:
            previous_questions = previous_questions[-10:]
        # Do something with the message here using LLM
        ai_message = await ai_response(message)

        return JsonResponse({'message': message, 'response': ai_message})
    return render(request, 'chatbot.html')