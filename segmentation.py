from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os

# Set the config and checkpoint file paths
config_file = r'C:\Users\tusha\OneDrive\Desktop\VW\mmsegmentation\configs\deeplabv3\deeplabv3_r50b-d8_4xb2-80k_cityscapes-769x769.py'
checkpoint_file = 'deeplabv3_r50b-d8_769x769_80k_cityscapes_20201225_155404-87fb0cf4.pth'

# Build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Input and output folder paths
input_folder = 'data/cityscapes/Q75/leftImg8bit/val/frankfurt/'  
output_folder = 'output_segmentation/Q75/leftImg8bit/val/frankfurt/' 

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of image file names in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

# Loop through the images and perform segmentation
for image_file in image_files:
    # Load and preprocess the image
    img = mmcv.imread(os.path.join(input_folder, image_file))

    # Perform semantic segmentation
    result = inference_model(model, img)

    # Save the visualization results to image files
    show_result_pyplot(model, img, result, show=False, out_file=os.path.join(output_folder, image_file),opacity=1)

print("Segmentation completed and results saved in", output_folder)
