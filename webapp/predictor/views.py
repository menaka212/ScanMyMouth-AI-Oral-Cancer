from django.shortcuts import render
from .utils import predict_and_gradcam
import os

def index(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        img_path = os.path.join('static', img.name)

        with open(img_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        result, confidence = predict_and_gradcam(img_path)

        # Risk level logic
        if confidence >= 80:
            risk = "HIGH"
        elif confidence >= 50:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        context = {
            'image_name': img.name,
            'result': result.upper(),
            'confidence': confidence,
            'risk': risk,
            'heatmap': True,
            'heatmap_url': '/media/heatmap.png'
        }

    return render(request, 'index.html', context)
