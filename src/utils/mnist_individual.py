import torch
from torchvision import datasets, transforms
from PIL import Image
import os

def extract_mnist_samples(output_dir="mnist_assets"):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load MNIST test dataset
    dataset = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.ToTensor())

    found_digits = set()
    
    print(f"Extracting digits to {output_dir}/...")
    
    for image, label in dataset:
        if label not in found_digits:
            # Convert tensor to PIL Image
            # MNIST tensors are [0, 1], we scale to [0, 255]
            img_array = (image.squeeze().numpy() * 255).astype('uint8')
            img = Image.fromarray(img_array, mode='L')
            
            # Save using the naming convention suggested for your LaTeX code
            file_name = f"digit{label}.png"
            img.save(os.path.join(output_dir, file_name))
            
            found_digits.add(label)
            print(f"Saved: {file_name}")
            
        if len(found_digits) == 10:
            break

    print("\nExtraction complete. You can now use these in your LaTeX 'mnist' macro.")

if __name__ == "__main__":
    extract_mnist_samples()