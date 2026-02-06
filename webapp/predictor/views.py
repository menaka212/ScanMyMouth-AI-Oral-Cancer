import os
from django.shortcuts import render
from django.conf import settings
from django.utils.text import get_valid_filename
from .utils import predict_and_gradcam


def index(request):
    return render(request, "index.html")


def analyzing(request):
    if request.method == "POST":

        image = request.FILES.get("image")
        if not image:
            return render(request, "index.html", {"error": "No image uploaded"})

        # Ensure media directory exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        # Sanitize filename
        safe_name = get_valid_filename(image.name)
        image_path = os.path.join(settings.MEDIA_ROOT, safe_name)

        # Save image
        with open(image_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        return render(request, "analyzing.html", {
            "image_name": safe_name
        })

    return render(request, "index.html")


def analyze(request):
    if request.method == "POST":

        image_name = request.POST.get("image_name")
        if not image_name:
            return render(request, "index.html", {"error": "Image missing"})

        image_path = os.path.join(settings.MEDIA_ROOT, image_name)

        result = predict_and_gradcam(image_path)

        return render(request, "result.html", result)

    return render(request, "index.html")
