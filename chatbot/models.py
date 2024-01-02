from django.db import models

# Create your models here.
class Chat(models.Model):
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    unique_id = models.TextField()

    def __str__(self):
        return f'{self.user.username}: {self.message}'