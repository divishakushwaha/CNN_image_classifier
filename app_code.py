import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from cnn_architecture import CIFAR_CNN

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    model = CIFAR_CNN()
    model.load_state_dict(torch.load('cifar_cnn.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Same pipeline as training — must match exactly
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the CNN will classify it into one of 10 categories: "
         "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence_pct = confidence.item() * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence_pct:.2f}%")

    # Show all class probabilities
    st.write("All probabilities:")
    for i, class_name in enumerate(CLASSES):
        st.write(f"{class_name}: {probabilities[0][i].item()*100:.2f}%")