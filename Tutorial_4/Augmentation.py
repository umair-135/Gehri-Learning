import os
import matplotlib.pyplot as plt         
from PIL import Image                  
from torchvision import transforms    
from torchvision.transforms import functional as TF


save_folder = r"/home/umair/Deep_Learning/Tutorial_4/Augmented_Images" 
os.makedirs(save_folder, exist_ok=True)

augment = transforms.Compose([
    transforms.RandomAffine(
        degrees=40,            
        shear=0.2,             
        scale=(0.8, 1.2),
    ),
    transforms.RandomHorizontalFlip(p=0.5),          
    transforms.ColorJitter(brightness=(0.5, 1.5)),   
])

img = Image.open(r"/home/umair/Deep_Learning/Tutorial_4/Original.jpeg").convert("RGB")  

num_images = 40

for i in range(num_images):
    augmented_image = augment(img)

    augmented_image.save(os.path.join(save_folder, f"image_{i}.jpeg"))


print(f"Augmented images are saved in the folder: {os.path.abspath(save_folder)}")