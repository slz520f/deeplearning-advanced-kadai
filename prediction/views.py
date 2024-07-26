from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

model = VGG16(weights='imagenet')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = request.FILES['image']
            img_bytes = BytesIO(img_file.read())
            img = load_img(img_bytes, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            
            result = model.predict(img_array)
            decoded_result = decode_predictions(result, top=5)[0]

            return render(request, 'result.html', {'decoded_result': decoded_result})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
