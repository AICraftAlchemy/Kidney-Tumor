from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import timm
import torch
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)

# Define class names
class_names = {
    0: "III_Malignant_RCC",
    1: "III_Malignant_Secondary",
    2: "II_Malignant_RCC",
    3: "II_Malignant_Secondary",
    4: "I_Benign_Adenoma",
    5: "I_Benign_Angiomyolipoma",
    6: "I_Benign_Lipomas",
    7: "I_Malignant_RCC",
    8: "I_Malignant_Secondary",
    9: "No_Tumor"
}


model_type = 'efficientnet-b0'  # You can choose other model types (b0 to b7)
model = EfficientNet.from_pretrained(model_type)
model.load_state_dict(torch.load('best_model_epoch_6.pt', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to process image and make predictions
# Function to process image and make predictions
def process_image(image):
    input_image = test_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)

    predicted_class = torch.argmax(output, dim=1).item()
    confidence = F.softmax(output, dim=1)[0][predicted_class].item()
    predicted_class_name = class_names[predicted_class]
    cure_info = get_cure_info(predicted_class_name)

    return predicted_class_name, confidence, cure_info, image


# Function to encode image as base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_img

# Function to get cure information for predicted class
def get_cure_info(predicted_class_name):
    # Define cure information
    cure_info = {
        "III_Malignant_RCC": "Regular screenings: Individuals with a family history of renal cell carcinoma (RCC) should undergo regular screenings and check-ups. Lifestyle modifications: Adopting a healthy lifestyle, including a balanced diet and regular exercise, can help reduce the risk of RCC. Avoid smoking: Smoking is a significant risk factor for RCC. Quitting smoking can lower the risk of developing this type of kidney cancer. Occupational safety: Workers exposed to certain chemicals, such as asbestos and cadmium, should take precautions to minimize exposure. Early detection: Awareness of symptoms such as blood in the urine, abdominal pain, and unexplained weight loss can facilitate early detection and treatment.",

        "III_Malignant_Secondary": "Monitor primary cancer: Patients with a history of primary cancer should undergo regular follow-up appointments and screenings to monitor for secondary tumors. Imaging studies: Regular imaging studies, such as CT scans and MRIs, may be necessary to detect secondary tumors in the kidneys. Genetic counseling: Individuals with a family history of secondary kidney tumors should consider genetic counseling and testing. Lifestyle modifications: Maintaining a healthy lifestyle, including a balanced diet and regular exercise, may reduce the risk of secondary kidney tumors. Chemoprevention: Some individuals at high risk of secondary kidney tumors may benefit from chemopreventive agents under the guidance of a healthcare professional.",

        "II_Malignant_RCC": "Regular check-ups: Individuals with a history of RCC or risk factors for kidney cancer should undergo regular check-ups and screenings. Imaging studies: Periodic imaging studies, such as ultrasound or CT scans, may be recommended to monitor for tumor recurrence. Lifestyle modifications: Adopting a healthy lifestyle, including smoking cessation and maintaining a healthy weight, can help reduce the risk of RCC recurrence. Medication adherence: Patients undergoing treatment for RCC should adhere to their prescribed medication regimen and follow-up appointments. Support groups: Joining support groups or seeking counseling can provide emotional support and guidance during the post-treatment period.",

        "II_Malignant_Secondary": "Early detection: Patients with a history of primary cancer should be vigilant about any new or unusual symptoms and report them to their healthcare provider promptly. Regular screenings: Individuals at high risk of developing secondary kidney tumors should undergo regular screenings and imaging studies. Genetic counseling: Individuals with a family history of secondary kidney tumors should consider genetic counseling to assess their risk and explore preventive measures. Lifestyle modifications: Maintaining a healthy lifestyle, including avoiding tobacco use and following a balanced diet, may help reduce the risk of secondary kidney tumors. Treatment adherence: Patients undergoing treatment for primary cancer should adhere to their prescribed treatment plan and attend follow-up appointments as recommended.",

        "I_Benign_Adenoma": "Regular monitoring: Individuals diagnosed with kidney adenomas should undergo regular monitoring with imaging studies to assess tumor growth and changes. Avoid unnecessary interventions: As kidney adenomas are typically benign and slow-growing, healthcare providers may recommend observation rather than immediate intervention. Imaging modalities: Various imaging modalities, such as ultrasound and MRI, may be used for surveillance of kidney adenomas. Lifestyle modifications: Maintaining a healthy lifestyle, including regular exercise and a balanced diet, may support overall kidney health. Consultation with specialists: Patients diagnosed with kidney adenomas may benefit from consultation with urologists or nephrologists to discuss management options and long-term care.",

        "I_Benign_Angiomyolipoma": "Regular monitoring: Patients diagnosed with angiomyolipomas should undergo regular monitoring with imaging studies to assess tumor growth and changes over time. Avoid trauma: Patients with large angiomyolipomas should avoid activities that may cause trauma to the kidneys, such as contact sports. Genetic counseling: Individuals with tuberous sclerosis complex (TSC) should receive genetic counseling and screening for associated kidney tumors, including angiomyolipomas. Lifestyle modifications: Maintaining a healthy lifestyle, including avoiding tobacco use and following a balanced diet, may help reduce the risk of complications associated with angiomyolipomas. Symptom awareness: Patients with angiomyolipomas should be aware of symptoms such as flank pain, hematuria, or abdominal mass and promptly report any changes to their healthcare provider.",

        "I_Benign_Lipomas": "Regular monitoring: Patients diagnosed with renal lipomas should undergo regular monitoring with imaging studies to assess tumor growth and changes over time. Imaging modalities: Various imaging modalities, such as ultrasound and CT scans, may be used for surveillance of renal lipomas. Lifestyle modifications: Maintaining a healthy lifestyle, including regular exercise and a balanced diet, may support overall kidney health. Symptom awareness: Patients with renal lipomas should be aware of symptoms such as flank pain or hematuria and promptly report any changes to their healthcare provider. Consultation with specialists: Patients diagnosed with renal lipomas may benefit from consultation with urologists or nephrologists to discuss management options and long-term care.",

        "I_Malignant_RCC": "Early detection: Individuals at high risk of developing RCC, such as those with a family history or certain genetic syndromes, should undergo regular screenings and check-ups. Imaging studies: Regular imaging studies, such as CT scans and MRIs, may be recommended for individuals at high risk of RCC to detect tumors at an early stage. Lifestyle modifications: Adopting a healthy lifestyle, including smoking cessation and maintaining a healthy weight, can help reduce the risk of RCC. Occupational safety: Workers exposed to certain chemicals, such as asbestos and cadmium, should take precautions to minimize exposure and reduce the risk of RCC. Genetic counseling: Individuals with a family history of RCC or certain genetic syndromes associated with kidney cancer should consider genetic counseling to assess their risk and explore preventive measures.",

        "I_Malignant_Secondary": "Regular monitoring: Patients with a history of primary cancer should undergo regular follow-up appointments and imaging studies to monitor for secondary tumors in the kidneys. Imaging studies: Regular imaging studies, such as CT scans and MRIs, may be necessary to detect secondary tumors in the kidneys. Genetic counseling: Individuals with a family history of secondary kidney tumors should consider genetic counseling and testing to assess their risk. Lifestyle modifications: Maintaining a healthy lifestyle, including avoiding tobacco use and following a balanced diet, may help reduce the risk of secondary kidney tumors. Chemoprevention: Some individuals at high risk of secondary kidney tumors may benefit from chemopreventive agents under the guidance of a healthcare professional.",

        "No_Tumor": "You are in safe zone"
           }

    return cure_info.get(predicted_class_name, "No cure information available.")




# Define Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            image = Image.open(file).convert('RGB')
            predicted_class, confidence, cure_info, original_image = process_image(image)
            encoded_image = encode_image(original_image)

            return render_template('index.html', predicted_class=predicted_class, confidence=confidence, cure_info=cure_info, encoded_image=encoded_image)

    return render_template('index.html')


