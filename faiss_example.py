import os
from dotenv import load_dotenv # Import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from reportlab.pdfgen import canvas


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set it in your .env file or as an environment variable.")

llm = OpenAI(temperature=0, api_key=api_key)

embeddings = OpenAIEmbeddings(api_key=api_key)

pdf_path = 'example_attention.pdf'

if not os.path.exists(pdf_path):
    
    print(f'Creating dummy PDF: {pdf_path}')
    
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, 'Attention Is All You Need')
    c.drawString(100, 730, 'Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,')
    c.drawString(100, 710, 'Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.')
    c.drawString(100, 690, 'Attention is a mechanism in neural networks that allows the model,')
    c.drawString(100, 670, 'to focus on relevant parts of the input. It is a function,')
    c.drawString(100, 650, 'that maps a query and a set of key-value pairs to an output.')
    c.drawString(100, 630, 'The transformer uses attention as a core component and completely dispenses with recurrence.')
    c.save()
else:
    print(f'Use existing PDF file: {pdf_path}')

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs) 
print(f'Number of text chunks created: {len(splits)}')

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
print('FAISS vector database successfully created.')

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(
    
llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

print('\n--- Start chat interaction ---')
user_question1 = 'What is attention?'
print(f'User: {user_question1}')
response1 = conversation_chain.invoke({'question': user_question1})
print('Bot:', response1['answer'])
user_question2 = 'Who is the author?'
print(f'\nUser: {user_question2}')
response2 = conversation_chain.invoke({'question': user_question2})
print('Bot:', response2['answer'])
print('\n--- Chat interaction ended ---')