import getpass
import os, ollama, time, json, numpy as np
import warnings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain.schema import AIMessage, HumanMessage
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Dict, List

from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
import pandas as pd
from langchain_core.prompts import PromptTemplate

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from numpy.linalg import norm
import openai, re
from pymongo import MongoClient
import uuid


os.environ['OPENAI_API_KEY']='api-key'
os.environ['LANGCHAIN_API_KEY']="lang-key"



llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)




#Misscelanoues functions


#Function to store the conversation data
def store_conversation_data( random_uuid, input_query, tool_called ):
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["conversations_db"]
    collection = db["user_data"]
    

    existing_user = collection.find_one({"user_id": random_uuid})

    if existing_user:
        # If the user exists, update the document by appending new data
        collection.update_one(
            {"user_id": random_uuid},
            {
                "$addToSet": {
                    "tool_called": {"$each": tool_called},
                    "input_query": {"$each": input_query}
                }
            }
        )
    else:
        # If the user doesn't exist, create a new user
        collection.insert_one({
            "user_id": random_uuid,
            "tool_called": tool_called,
            "input_query": input_query
        })

    print(f"Data has been successfully added/updated for user: {random_uuid}")




def total_assembly(in_dict: Dict[str, str])->str:

    in_dict_keys=list(in_dict.keys())
    in_dict_values=list(in_dict.values())
    out_html_list=["""<div class="container text-center mt-4"> """,
               """<p class="fw-bold fs-5 text-primary">Following are the selected components</p>""",
               """<ol class="list-group list-group-numbered mx-auto mt-1" style="max-width: 450px;">""",
     
               ]
    val_list=[]
    for ky, val in zip(in_dict_keys, in_dict_values):

        out_html_list.append( f""" <li class="list-group-item d-flex  align-items-center"> {ky.upper()}: {val.split('@')[0].strip()}  </li>"""  )
        val_list.append(int(val.split("@")[1].replace("$",'').strip()))

    tail_html_list=[
        "</ol>",
        f"""<p class="fw-bold fs-5 text-primary">Grand Total ${sum(val_list)}</p>""",
        
        "</div>"
              
       ]

    final_html_list=out_html_list+tail_html_list
    final_html_str="".join(final_html_list)
    
    return(final_html_str)



#Code Block for RAG Function
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)

    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]

    # print(needle, haystack[0])
    # print(np.dot(needle, haystack[0]))
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def rag_function(prompt, embedding_model="mxbai-embed-large:latest", filename='laptop_data.txt'):

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Do not give answers that are outside the context given.
        Answer in about 50 words for each query. Transform the input text in your own way to provide concise answers.
        Context:
    """

    
    
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, embedding_model, paragraphs)



    
    prompt_embedding = ollama.embeddings(model=embedding_model, prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    openai.api_key='api-key'

    chat_completion = openai.chat.completions.create(
    model="gpt-3.5-turbo-0125", temperature=0.7,
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT + "\n".join(paragraphs[item[1]] for item in most_similar_chunks)  },
        {"role": "user", "content": prompt}
    ],

        )

   

    # print("\n\n")
    # print(chat_completion.choices[0].message.content)
    return(chat_completion.choices[0].message.content)


#Ingest all the data
df_laptops=pd.read_excel('laptops.xlsx', dtype={'PRICE': int, 'RAM(GB)': int, 'STORAGE(GB)': int, 'RATINGS': float})
df_laptops=df_laptops[['id', 'PRODUCT', 'BRAND', 'PRICE', 'RAM(GB)', 'STORAGE(GB)','PROCESSOR', 'RATINGS']]
# df_CPU=pd.read_excel('CPU.xlsx')
# df_GPU=pd.read_excel('GPU.xlsx')
# df_monitor=pd.read_excel('monitor.xlsx')


#List of all the prompts

greet_prompt=PromptTemplate(
            input_variables=["query"],
            template="""                  
                  Your a very helpful assistant for a Ecommerce store, that sells computer and related accessories online.
                  If the user greets, greet back. Your name is Maya.
                  If the user expresses gratitude, give an appropriate response.
                  If your name is asked, answer your name politely. If the user provides his/her name, use the name in your replies.
                  Query: {query}
                  
                  """
       )



def format_string(in_str:str)->str:
   
    in_str_split=re.split(r'(?=\n\d+\.\s)', in_str)
    card_pattern=r'^\d+\..*'
    format_list=["""<div class= "card-body">"""]

    for ele in in_str_split:
        ele=ele.replace("\n", '')
    
        if re.search(card_pattern, ele):
            ele= f"""<div class= "card-content"> {ele}  </div>"""
        else:
            ele= f"""<div class= "card-title"> {ele}  </div>"""
            
        format_list.append(ele)
        
    format_list.append("</div>")
    format_str= "".join(format_list)
          
          
    return(format_str)





@tool
def bot_greet(user_input:str)-> str:
    """Tool to be called when user greets, expresses gratitude, asks your name"""

    # print("greet called")
    greet_llm_chain= greet_prompt | llm

    response=greet_llm_chain.invoke(user_input)
    
    

    try:
        out_dict ={
                  'response':response.content,
                  'html':'no'
            
                  }
        return(out_dict)
    
    except:
        out_dict ={
                  'response':'I am your Ecommerce Assistant, How may I assist you today?',
                  'html':'no'
            
                  }

        
        return(out_dict)



@tool
def bot_laptop(user_input:str):
    """Function to be called, when user inquires about computers, laptops, prices, ratings, brands"""
    agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
            df_laptops,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True
        )

    agent_str=agent.invoke(user_input)['output']
    formatted_str=format_string(agent_str)
    
    out_dict ={
                  'response':formatted_str,
                  'html':'yes'
            
                  }
   
    

    return(out_dict)




@tool
def bot_general_info(user_input:str):
    """Function to be called, when user inquires about the store, reviews of different brands, product types and services provided, GPUS and their comparisons,
     Payment options, Return Policy"""

    response=rag_function(user_input)
    
    out_dict ={
                  'response':response,
                  'html':'yes'
            
                  }
 

    return(out_dict)


@tool
def bot_assembly(user_input:str):
    """Function to be called, when user wants/requests to guide him in assembling or gathering individual components of computer/laptop"""

 
    out_dict ={
                  'response':'Alright, Lets start with processors, pick one',
                   'html':'assemble_chain'
            
                  }
    

    return(out_dict)





def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in ecom_tools}
   
    tool_calls = msg.tool_calls.copy()

    
    #Introducing the Fall back statement
    if len(tool_calls)==0:
        print("========================Fallback Action=====================  \n")
        print("I am the Food Assistant Alexa, I am here to assist you in ordering your delicious meal!")
    else:
        for tool_call in tool_calls:

           tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
           
           if tool_call['name']:
               tool_called=[tool_call['name']]
           else:
               tool_called=['None']
               
            
           return(tool_call["output"], tool_called) 
       
       
       
ecom_tools = [bot_greet, bot_laptop, bot_general_info, bot_assembly]
llm_with_tools =  llm.bind_tools(ecom_tools) 
chain = llm_with_tools | call_tools






