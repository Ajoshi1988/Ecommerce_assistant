from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin



urlpatterns = [
    
    path('', views.home, name="home"),
    path('register', views.register, name="register"),
    path('my-login', views.my_login, name="my-login"),
    
    path('user-logout', views.user_logout, name="user-logout" ),
    
    path('profile', views.profile, name="profile" ),
    
    path('password-reset-custom', views.PasswordReset, name="password_reset" ),
 
    # path('password-reset/',
    #      auth_views.PasswordResetView.as_view(
    #          template_name='account/password_reset.html'
    #      ),
    #      name='password_reset'),
    
      

    
    # path('password-reset/done/',
    #      auth_views.PasswordResetDoneView.as_view(
    #          template_name='account/password_reset_done.html'
    #      ),
    #      name='password_reset_done'),
    
    # path('password-reset-confirm/<uidb64>/<token>/',
    #      auth_views.PasswordResetConfirmView.as_view(
    #          template_name='account/password_reset_confirm.html'
    #      ),
    #      name='password_reset_confirm'),
    
    # path('password-reset-complete/',
    #      auth_views.PasswordResetCompleteView.as_view(
    #          template_name='account/password_reset_complete.html'
    #      ),
    #      name='password_reset_complete'),
   
    
   
]
