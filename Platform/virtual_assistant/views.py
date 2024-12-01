from django.shortcuts import render, HttpResponse, get_object_or_404
from .models import *
from django.contrib.auth.decorators import login_required
from .forms import *
from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.urls import reverse
import re
# Create your views here.

from .html_content import htmls
from .bot import chain, total_assembly, store_conversation_data
from django.core.cache import cache
from threading import Lock
import uuid
import pandas as pd
import os

lock = Lock()



cache.set("uuid",'')
cache.set("order_dict",{})

def VirtualAssistant(request):
    
    return render(request, 'virtual_assistant/vi.html')



def ChatApp(request):
    chat_group=get_object_or_404(ChatGroup, group_name="public-chat")
    chat_messages=chat_group.chat_messages.all()[:30]
    
    form=ChatmessageCreateForm()
    
    if request.method=='POST':
        form=ChatmessageCreateForm(request.POST)
        if form.is_valid:
            message=form.save(commit=False)
            message.author=request.user
            message.group=chat_group
            message.save()
            
            return redirect('home')
    
    return render(request, 'virtual_assistant/chat_application.html', {'chat_messages':chat_messages, 'form':form })
    
    
    
def MLApp(request):
    
    return render(request, 'virtual_assistant/machine_learning.html')
    
    
    
def OCR(request):
    
    return render(request, 'virtual_assistant/OCR.html')
 
 
def ChartsJs(request):
    
    return render(request, 'virtual_assistant/dashboard.html')
 


##########################################################



def ChatBot(request):
    
    return render(request, 'virtual_assistant/chat_bot.html')
 




def bot_response(request):
    
    bot_query = request.GET.get('user_query', None)
    
    print(f"bot Query is {bot_query}")

       
    
    chain_response, tool_called=chain.invoke(str(bot_query))
    
    # Store the conversation data
    if cache.get("uuid"):
        random_uuid=cache.get("uuid")
        print(random_uuid)
    else:
        random_uuid =str(uuid.uuid4())
        cache.set("uuid", random_uuid)
        print(random_uuid)
        
    store_conversation_data( random_uuid, [bot_query], tool_called )
        
    

    response_data = {
        'message': chain_response['response'],
        
        'html': chain_response['html']
        
        
    }

    return JsonResponse(response_data)

       

def data_from_user(request):
    
    with lock:
    
        button_class = request.GET.get('button_class', None)
        button_text = request.GET.get('button_text', None)
        
        print(f"data from user {button_class, button_text}")
        
        if cache.get('order_dict'):
            order_dict=cache.get('order_dict')
        else:
            order_dict={}
            
        
        
        if re.findall(r'\bCPU\b', button_class):
            
            order_dict['CPU']= button_text
            cache.set('order_dict',order_dict, timeout=900)
           
            message="Next, pick one among the GPUs"
            html="GPU_buttons"

        if re.findall(r'\bGPU\b', button_class):
            
            order_dict['GPU']= button_text
            cache.set('order_dict',order_dict, timeout=900)

            
            message="Select the SSD of your choice."
            html="SSD_buttons"

        if re.findall(r'\bSSD\b', button_class):
            
            order_dict['SSD']= button_text
            cache.set('order_dict',order_dict, timeout=900)

            
            message="Lastly, select the monitor of your choice"
            html="monitor_buttons"
            
            
        if re.findall(r'\bmonitor\b', button_class):
            
            order_dict['monitor']= button_text
            cache.set('order_dict',order_dict, timeout=900)
            
            final_order_dict=cache.get("order_dict")
            final_order_html=total_assembly(final_order_dict)

            
            message=final_order_html
            
            html="book_buttons"  
            
            
        if  (button_text=='Continue'): 
            message="Great, Here are your payment options"
            html="pay_continue"
 
        if  (button_text=='No'): 
            message="Cool, Continue exploring, Ask any query 😃😃"
            html="pay_no"
            
        if re.findall(r'\bpay_money\b', button_class):
            message=f"Alright, A notification will be sent on your {button_text} app to complete the payment process, Thank you for shopping with us 🙏🙏"
            html="pay_link"
            
            # Save the purchase order
            random_uuid=cache.get("uuid")
            order_dict=cache.get("order_dict")
            
            
            file_path = "Purchases.csv"
            order_save_dict={'buyer_id': [random_uuid], 
                             'CPU': [order_dict['CPU']],
                             'GPU': [order_dict['GPU']],
                             'SSD': [order_dict['SSD']],
                             'Monitor': [order_dict['monitor']],
                                                          
                             }
            if not os.path.exists(file_path):
                pd.DataFrame(order_save_dict).to_csv(file_path, mode='w', index=False, header=True)
            else:
                pd.DataFrame(order_save_dict).to_csv(file_path, mode='a', index=False, header=True)
                
            
            
            cache.set("order_dict", {})
            
            
            
            
            
        print("Accumulated dictis", cache.get('order_dict'))  
            
            
        
        response_data = {
            'message': message,
            
            'html': "assemble_chain",
            
            'next': html,
            
            
            
        }
        return JsonResponse(response_data)
    
    







