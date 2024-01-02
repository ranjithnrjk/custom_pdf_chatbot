from django.db import models
from django.utils import timezone

class Chat(models.Model):
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    unique_id = models.TextField()

    def __str__(self):
        return f'{self.created_at}: {self.message}'
