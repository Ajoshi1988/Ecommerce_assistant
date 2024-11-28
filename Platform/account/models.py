from django.db import models
from .managers import CustomUserManager
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils import timezone
from PIL import Image
from django.conf import settings



class CustomUser(AbstractBaseUser, PermissionsMixin):
    
    username=None
    email=models.EmailField(unique=True)
    first_name=models.CharField(max_length=80)
    last_name=models.CharField(max_length=135)
    is_active=models.BooleanField(default=True)
    is_staff=models.BooleanField(default=True)
    date_joined=models.DateTimeField(default=timezone.now)
    
    # is_writer=models.BooleanField(default=False, verbose_name="are you a writer?")
    
    
    USERNAME_FIELD='email'
    REQUIRED_FIELDS=[]
    
    objects=CustomUserManager()
    
    def __str__(self):
        
        return self.email
    
  
    
#Profile class for models    
class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='images')
    
    def __str__(self):
        return f'{self.user.first_name} Profile'
    
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        img = Image.open(self.image.path)

        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)

    
    