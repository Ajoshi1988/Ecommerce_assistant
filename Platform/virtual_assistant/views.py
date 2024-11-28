from django.shortcuts import render, HttpResponse, get_object_or_404
from .models import *
from django.contrib.auth.decorators import login_required
from .forms import *
from django.shortcuts import redirect, render
from django.http import JsonResponse

# Create your views here.

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

    print("Bot called")

    response_data = {
        'message': 'Hello from Django!',
        'status': 'success',
    }

    return JsonResponse(response_data)

       

    