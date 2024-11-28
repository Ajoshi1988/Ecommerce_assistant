from django.contrib.auth.forms import UserCreationForm
from . models import CustomUser, Profile
from django import forms

from django.core.mail import send_mail
from django.contrib.auth.forms import PasswordResetForm
from django.conf import settings


class CreateUserForm(UserCreationForm):
    
    class Meta:
        model=CustomUser
        fields=['email','first_name', 'last_name', 'password1', 'password2']
        
        
        
        
class UserUpdateForm(forms.ModelForm):
    
    class Meta:
        model = CustomUser
        fields = ['first_name', 'last_name']


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['image']
        
        
               
        


