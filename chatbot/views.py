from .models import Chat
from django.utils import timezone
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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import uuid
from asgiref.sync import sync_to_async

unique_id = str(uuid.uuid4())

device = torch.device('cpu')
# checkpoint = "MBZUAI/LaMini-GPT-1.5B" 
# checkpoint = 'Salesforce/codet5p-16b'
# checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
checkpoint = "MBZUAI/LaMini-Cerebras-1.3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=512, use_auth_token=True)
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
based on the context provided and user previous questions. Along with your knowledge use the context provided and user previous questions to give answer to the question. 

> If you really don't know the answer, say "I'm sorry, I don't know the answer to that." 
and then direct the questioner to contact help@company.com for human assitance. 

> Always speak as if you were chatting to a friend. Keep the answer simple and to the point.

> Answer with atmost 100 words.

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
    
@sync_to_async
def get_previous_questions(uniqueId):
    # Retrieve chat messages with a specific unique_id
    filtered_chats = Chat.objects.filter(unique_id=uniqueId)
    previous_questions = []
    # Iterate through the queryset and print the values
    for chat in filtered_chats:
        previous_questions.append(chat.message)
    return previous_questions

async def ai_response(message):
    # Do something with the message here using LLM
    retriever_response = await util1(message)

    previous_questions = await get_previous_questions(unique_id)

    if len(retriever_response) > 512:
        retriever_response = retriever_response[:512]
    ai_message =  answerChain.invoke({'context': retriever_response, 
                                      'question': message,
                                      'previous_questions': previous_questions})
    return ai_message

@sync_to_async
def database_saver(message, ai_message):
    chat_instance = Chat(
    message=message,
    response=ai_message,
    created_at=timezone.now(),  # This field is auto-populated, so you don't need to provide a value
    unique_id=unique_id
    )

    # Save the instance to the database
    chat_instance.save()

# Create your views here.
async def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        # Do something with the message here using LLM
        ai_message = await ai_response(message)

        await database_saver(message, ai_message)

        return JsonResponse({'message': message, 'response': ai_message})
    return render(request, 'chatbot.html')