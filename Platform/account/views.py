from django.shortcuts import render, redirect
from . forms import CreateUserForm, UserUpdateForm, ProfileUpdateForm

from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponse

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import user_passes_test
from django.contrib import messages
from django.template import RequestContext
from django.contrib.auth.decorators import login_required

from django.contrib.auth.views import PasswordResetView

from django.urls import reverse_lazy


def home(request):
    
    return render(request, 'account/index.html')
    


def register(request):
    
    
    #allowed to register, only if the user is superusr or the admin.
    
    if request.user.is_superuser:
        form=CreateUserForm()
        
        if request.method=='POST':
            form=CreateUserForm(request.POST)
            
            if form.is_valid():
                
                form.save()
                
                #save Profile
                
                
                return redirect('my-login')
            
            
                
        context={'RegisterForm':form}    
            
       
        return render(request, 'account/register.html', context)
    
    else:
        
        messages.error(request, 'Only Admin users are allowed to Register new users!!')
        return redirect( 'home')




def my_login(request):
    
    #If already logged in then redirect to landing page, otherwise perfomr nauithentication
    
    if request.user.is_authenticated:
        
        return redirect('landing_page')
    
    else:
          
    
        form=AuthenticationForm()
        
        if request.method=='POST':
            form=AuthenticationForm(request, data=request.POST)
            
            if form.is_valid():
                
                username=request.POST.get('username')
                password=request.POST.get('password')
                
                user=authenticate(request, username=username, password=password)
                
                if user is not None:
                    login(request, user)
                    
                    return redirect('landing_page')
                
                
                
                
        context={'LoginForm':form}
            
        
        
        return render(request, 'account/my-login.html', context)




def user_logout(request):
    logout(request)
    
    messages.success(request, 'You are Logged out now')
    return redirect("home")





@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST,
                                    request.FILES,
                                    instance=request.user.profile)
        
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render(request, 'account/profile.html', context)





@login_required
def PasswordReset(request):
    
    if request.method == 'POST':
        
        current_password=request.POST.get("current_password_name")
        new_password=request.POST.get("new_password_name")
        confirm_password=request.POST.get("confirm_password_name")
        
        user = request.user
        
        if user.check_password(current_password):
            
            if (new_password!=confirm_password) and (new_password!=None) and (confirm_password!=None):
                
                messages.error(request, f'New Password and Current Password do not match')
                
                return render(request, 'account/password_reset.html')
                
            else:
                               
                user.set_password(new_password)
                user.save()
                
                messages.success(request, f'Password reset successfully')
                
                return redirect("home")
                   
               
         
        else:
            messages.error(request, f'Current Password is incorrect!!')
               
    
            return render(request, 'account/password_reset.html')
    
    return render(request, 'account/password_reset.html')