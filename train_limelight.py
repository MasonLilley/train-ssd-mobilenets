# Limelight Neural Detector Training Notebook - Docker Version
# Designed for local Docker environment only

import shutil
import os
import sys
import re

# ============================================================================
# 1. Environment Setup
# ============================================================================

print("="*80)
print("LIMELIGHT NEURAL DETECTOR TRAINING - DOCKER ENVIRONMENT")
print("="*80)
print(f"Python version: {sys.version}")

# TRAINING PARAMETERS!
num_steps = 40000
checkpoint_every = 2500
batch_size = 32

# Set paths for Docker environment
HOMEFOLDER = '/workspace/'
FINALOUTPUTFOLDER = '/workspace/final_output'
DATASET_PATH = '/workspace/data/dataset.zip'

print(f"Home folder: {HOMEFOLDER}")
print(f"Output folder: {FINALOUTPUTFOLDER}")
print(f"Dataset path: {DATASET_PATH}")

# Verify dataset exists
if not os.path.exists(DATASET_PATH):
    print("\n" + "!"*80)
    print("ERROR: Dataset not found!")
    print(f"Please place your dataset.zip file at: {DATASET_PATH}")
    print("Mount it when running docker with: -v $(pwd)/data:/workspace/data")
    print("!"*80)
    sys.exit(1)

print("✓ Dataset found")

# Verify models repository exists (should be installed by Dockerfile)
if not os.path.exists(f'{HOMEFOLDER}models'):
    print("\n" + "!"*80)
    print("ERROR: TensorFlow models repository not found!")
    print("This should have been installed by the Dockerfile.")
    print("Please rebuild your Docker image.")
    print("!"*80)
    sys.exit(1)

print("✓ TensorFlow models repository found")
print("✓ Environment setup complete\n")

# ============================================================================
# 2. Load and Extract Dataset
# ============================================================================

print("="*80)
print("DATASET EXTRACTION")
print("="*80)

# Create a dedicated directory for the dataset
DATASET_DIR = f'{HOMEFOLDER}dataset/'
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

print(f"Extracting {DATASET_PATH}...")
extract_result = os.system(f'unzip -q {DATASET_PATH} -d {DATASET_DIR}')
if extract_result != 0:
    print("\n" + "!"*80)
    print("ERROR: Failed to extract dataset!")
    print(f"Please verify {DATASET_PATH} is a valid zip file")
    print("!"*80)
    sys.exit(1)
print(f"✓ Dataset extracted to {DATASET_DIR}\n")

# Auto-detect tfrecord files
import fnmatch

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield os.path.join(root, basename)

def find_tfrecord_files(directory):
    train_record = ''
    val_record = ''
    label_map = ''
    
    for tfrecord_file in find_files(directory, '*.tfrecord'):
        if '/train/' in tfrecord_file:
            train_record = tfrecord_file
        elif '/valid/' in tfrecord_file:
            val_record = tfrecord_file
    
    # Find label map - prioritize train/valid/test folders over models folder
    for label_file in find_files(directory, '*_label_map.pbtxt'):
        # Skip the MSCOCO label map in the models folder
        if '/models/research/object_detection/data/' not in label_file:
            label_map = label_file
            break
    
    return train_record, val_record, label_map

# Search in both HOMEFOLDER and HOMEFOLDER/data for flexibility
train_record_fname, val_record_fname, label_map_pbtxt_fname = find_tfrecord_files(DATASET_DIR)

# If not found, error out with helpful message
if not train_record_fname or not val_record_fname or not label_map_pbtxt_fname:
    print("\n" + "!"*80)
    print("ERROR: Could not find required dataset files!")
    print(f"Train record: {train_record_fname or 'NOT FOUND'}")
    print(f"Validation record: {val_record_fname or 'NOT FOUND'}")
    print(f"Label map: {label_map_pbtxt_fname or 'NOT FOUND'}")
    print("\nPlease verify your dataset.zip contains:")
    print("  - A 'train' folder with .tfrecord files")
    print("  - A 'valid' folder with .tfrecord files")
    print("  - A label map .pbtxt file")
    print("!"*80)
    sys.exit(1)

print("✓ Found train record:", train_record_fname)
print("✓ Found validation record:", val_record_fname)
print("✓ Found label map:", label_map_pbtxt_fname)
print()

# ============================================================================
# 3. Download Pre-trained Model and Configuration
# ============================================================================

print("="*80)
print("MODEL CONFIGURATION")
print("="*80)

chosen_model = 'ssd-mobilenet-v2'
MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'limelight_ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
}

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

print(f"Selected model: {chosen_model}")
print(f"Model name: {model_name}")

# Create model directory
mymodel_dir = f'{HOMEFOLDER}models/mymodel/'
os.makedirs(mymodel_dir, exist_ok=True)
os.chdir(mymodel_dir)

# Download pre-trained model weights
import tarfile
download_tar = f'https://downloads.limelightvision.io/models/{pretrained_checkpoint}'
print(f"\nDownloading pre-trained weights...")
download_result = os.system(f'wget -q {download_tar}')
if download_result != 0:
    print("\n" + "!"*80)
    print("ERROR: Failed to download pre-trained weights!")
    print(f"URL: {download_tar}")
    print("Please check your internet connection")
    print("!"*80)
    sys.exit(1)
print("✓ Downloaded")

print("Extracting weights...")
try:
    tar = tarfile.open(pretrained_checkpoint)
    tar.extractall()
    tar.close()
    print("✓ Extracted")
except Exception as e:
    print("\n" + "!"*80)
    print("ERROR: Failed to extract pre-trained weights!")
    print(f"Error: {str(e)}")
    print("!"*80)
    sys.exit(1)

# Download training configuration
download_config = f'https://downloads.limelightvision.io/models/{base_pipeline_file}'
print(f"Downloading base configuration...")
config_result = os.system(f'wget -q {download_config}')
if config_result != 0:
    print("\n" + "!"*80)
    print("ERROR: Failed to download configuration file!")
    print(f"URL: {download_config}")
    print("Please check your internet connection")
    print("!"*80)
    sys.exit(1)
print("✓ Downloaded")

os.chdir(HOMEFOLDER)

print(f"\n✓ Training parameters:")
print(f"  Total steps: {num_steps}")
print(f"  Checkpoint every: {checkpoint_every} steps")
print(f"  Batch size: {batch_size}")
print()

# ============================================================================
# 4. Generate Labels File
# ============================================================================

print("="*80)
print("LABEL GENERATION")
print("="*80)

pipeline_fname = f'{mymodel_dir}{base_pipeline_file}'
fine_tune_checkpoint = f'{mymodel_dir}{model_name}/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    try:
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())
    except Exception as e:
        print("\n" + "!"*80)
        print("ERROR: Failed to parse label map!")
        print(f"Error: {str(e)}")
        print("!"*80)
        sys.exit(1)

def get_classes(pbtxt_fname):
    try:
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return [category['name'] for category in category_index.values()]
    except Exception as e:
        print("\n" + "!"*80)
        print("ERROR: Failed to parse label map!")
        print(f"Error: {str(e)}")
        print("!"*80)
        sys.exit(1)

def create_label_file(filename, labels):
    with open(filename, 'w') as file:
        for label in labels:
            file.write(label + '\n')

num_classes = get_num_classes(label_map_pbtxt_fname)
classes = get_classes(label_map_pbtxt_fname)

print(f"✓ Total classes: {num_classes}")
print(f"✓ Classes: {classes}")

# Generate labels file
labels_file = f"{HOMEFOLDER}limelight_neural_detector_labels.txt"
create_label_file(labels_file, classes)
print(f"✓ Labels file created: {labels_file}")
print()

# ============================================================================
# 5. Create Custom Pipeline Configuration
# ============================================================================

print("="*80)
print("PIPELINE CONFIGURATION")
print("="*80)

print("Creating custom pipeline configuration...")

try:
    with open(pipeline_fname) as f:
        s = f.read()
except FileNotFoundError:
    print("\n" + "!"*80)
    print("ERROR: Pipeline configuration file not found!")
    print(f"Expected location: {pipeline_fname}")
    print("!"*80)
    sys.exit(1)

try:
    with open(f'{HOMEFOLDER}pipeline_file.config', 'w') as f:
        # Set fine_tune_checkpoint path
        s = re.sub('fine_tune_checkpoint: ".*?"',
                   f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
        
        # Set tfrecord files for train and test datasets
        s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                   f'input_path: "{train_record_fname}"', s)
        s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                   f'input_path: "{val_record_fname}"', s)
        
        # Set label_map_path
        s = re.sub('label_map_path: ".*?"', 
                   f'label_map_path: "{label_map_pbtxt_fname}"', s)
        
        # Set batch_size
        s = re.sub('batch_size: [0-9]+', f'batch_size: {batch_size}', s)
        
        # Set training steps
        s = re.sub('num_steps: [0-9]+', f'num_steps: {num_steps}', s)
        
        # Set number of classes
        s = re.sub('checkpoint_every_n: [0-9]+', f'num_classes: {num_classes}', s)
        
        # Change fine-tune checkpoint type from "classification" to "detection"
        s = re.sub('fine_tune_checkpoint_type: "classification"', 
                   'fine_tune_checkpoint_type: "detection"', s)
        
        # Adjust learning rate for ssd-mobilenet-v2
        if chosen_model == 'ssd-mobilenet-v2':
            s = re.sub('learning_rate_base: .8', 'learning_rate_base: .004', s)
            s = re.sub('warmup_learning_rate: 0.13333', 'warmup_learning_rate: .0016666', s)
        
        f.write(s)
except Exception as e:
    print("\n" + "!"*80)
    print("ERROR: Failed to write pipeline configuration!")
    print(f"Error: {str(e)}")
    print("!"*80)
    sys.exit(1)

pipeline_file = f'{HOMEFOLDER}pipeline_file.config'
model_dir = f'{HOMEFOLDER}training_progress/'

print(f"✓ Pipeline configuration saved: {pipeline_file}")
print(f"✓ Training checkpoints will be saved to: {model_dir}")
print()

# ============================================================================
# 6. Train Model
# ============================================================================

print("="*80)
print("TRAINING")
print("="*80)
print()
print("To monitor training progress, open a new terminal and run:")
print("  docker exec -it <container_id> tensorboard --logdir /workspace/training_progress/train --host 0.0.0.0 --port 6006")
print("Then visit http://localhost:6006 in your browser")
print()
print("Or run this in a separate notebook cell:")
print("  !tensorboard --logdir /workspace/training_progress/train --host 0.0.0.0 --port 6006 &")
print()
print("="*80)
print("STARTING TRAINING - This will take several hours")
print("="*80)
print()

# Clean up any previous training
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# Run training
train_command = f"""python {HOMEFOLDER}models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_file} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --checkpoint_every_n={checkpoint_every} \
    --num_train_steps={num_steps} \
    --num_workers=2 \
    --sample_1_of_n_eval_examples=1"""

train_result = os.system(train_command)

if train_result != 0:
    print("\n" + "!"*80)
    print("ERROR: Training failed!")
    print("Please check the error messages above")
    print("!"*80)
    sys.exit(1)

print()
print("="*80)
print("TRAINING COMPLETE")
print("="*80)
print()

# ============================================================================
# 7. Convert Model to TFLite
# ============================================================================

print("="*80)
print("TFLITE CONVERSION")
print("="*80)

# Remove final output folder if it exists
if os.path.exists(FINALOUTPUTFOLDER):
    shutil.rmtree(FINALOUTPUTFOLDER)

# Create output directory
os.makedirs(FINALOUTPUTFOLDER, exist_ok=True)
print(f"✓ Output directory created: {FINALOUTPUTFOLDER}")

# Export graph
last_model_path = f'{HOMEFOLDER}training_progress'
exporter_path = f'{HOMEFOLDER}models/research/object_detection/export_tflite_graph_tf2.py'

# Verify training checkpoint exists
if not os.path.exists(last_model_path) or not os.listdir(last_model_path):
    print("\n" + "!"*80)
    print("ERROR: No training checkpoints found!")
    print(f"Expected location: {last_model_path}")
    print("Training may have failed")
    print("!"*80)
    sys.exit(1)

print("Exporting TFLite graph...")
export_command = f"""python {exporter_path} \
    --trained_checkpoint_dir {last_model_path} \
    --output_directory {FINALOUTPUTFOLDER} \
    --pipeline_config_path {pipeline_file}"""
export_result = os.system(export_command)

if export_result != 0:
    print("\n" + "!"*80)
    print("ERROR: Failed to export TFLite graph!")
    print("!"*80)
    sys.exit(1)
print("✓ Graph exported")

# Convert to .tflite
print("Converting to TFLite format...")
import tensorflow as tf

try:
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{FINALOUTPUTFOLDER}/saved_model')
    tflite_model = converter.convert()
    model_path_32bit = f'{FINALOUTPUTFOLDER}/limelight_neural_detector_32bit.tflite'
    
    with open(model_path_32bit, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ 32-bit TFLite model saved: {model_path_32bit}")
except Exception as e:
    print("\n" + "!"*80)
    print("ERROR: Failed to convert to TFLite!")
    print(f"Error: {str(e)}")
    print("!"*80)
    sys.exit(1)

# Copy labels and config to output
shutil.copy(labels_file, FINALOUTPUTFOLDER)
shutil.copy(pipeline_file, FINALOUTPUTFOLDER)
print("✓ Copied labels and configuration files")
print()

# ============================================================================
# 8. Quantize Model
# ============================================================================

print("="*80)
print("QUANTIZATION")
print("="*80)

import tensorflow as tf
import io
from PIL import Image
import glob
import random

def extract_images_from_tfrecord(tfrecord_path, output_folder, num_samples=100):
    """Extract sample images from tfrecord for quantization"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    saved_images = 0
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    for raw_record in raw_dataset.take(num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        image_data = example.features.feature['image/encoded'].bytes_list.value[0]
        image = Image.open(io.BytesIO(image_data))
        image.save(os.path.join(output_folder, f'image_{saved_images}.png'))
        saved_images += 1
        if saved_images >= num_samples:
            break
    
    return saved_images

extracted_sample_folder = f'{HOMEFOLDER}extracted_samples'

# Remove sample folder if it exists
if os.path.exists(extracted_sample_folder):
    shutil.rmtree(extracted_sample_folder)

# Extract images
print("Extracting sample images for quantization...")
num_extracted = extract_images_from_tfrecord(train_record_fname, extracted_sample_folder)
print(f"✓ Extracted {num_extracted} images to {extracted_sample_folder}")

# Get list of all images
image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png', '*.bmp']
quant_image_list = []
for ext in image_extensions:
    quant_image_list.extend(glob.glob(f'{extracted_sample_folder}/{ext}'))

print(f"✓ Found {len(quant_image_list)} images for quantization")

# Get input dimensions from the 32-bit model
try:
    interpreter = tf.lite.Interpreter(model_path=model_path_32bit)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print(f"✓ Model input size: {width}x{height}")
except Exception as e:
    print("\n" + "!"*80)
    print("ERROR: Failed to load 32-bit model for quantization!")
    print(f"Error: {str(e)}")
    print("!"*80)
    sys.exit(1)

# Representative dataset generator
def representative_data_gen():
    """Generate representative dataset for quantization"""
    dataset_list = quant_image_list
    quant_num = min(300, len(dataset_list))
    
    print(f"Processing {quant_num} images for quantization...")
    for i in range(quant_num):
        pick_me = random.choice(dataset_list)
        if i % 50 == 0:
            print(f"  Processing image {i}/{quant_num}")
        
        image = tf.io.read_file(pick_me)
        
        # Decode based on file extension
        if pick_me.endswith(('.jpg', '.JPG', '.jpeg')):
            image = tf.io.decode_jpeg(image, channels=3)
        elif pick_me.endswith('.png'):
            image = tf.io.decode_png(image, channels=3)
        elif pick_me.endswith('.bmp'):
            image = tf.io.decode_bmp(image, channels=3)
        
        image = tf.image.resize(image, [width, height])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

# Initialize converter for INT8 quantization
print("\nInitializing INT8 quantization converter...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{FINALOUTPUTFOLDER}/saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    
    print("✓ Converter initialized")
    print("\nBeginning conversion (this may take several minutes)...")
    tflite_model = converter.convert()
    print("✓ Conversion complete!")
    
    quantized_model_path = f'{FINALOUTPUTFOLDER}/limelight_neural_detector_8bit.tflite'
    with open(quantized_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ 8-bit quantized model saved: {quantized_model_path}")
except Exception as e:
    print("\n" + "!"*80)
    print("ERROR: Quantization failed!")
    print(f"Error: {str(e)}")
    print("!"*80)
    sys.exit(1)
print()

# ============================================================================
# 9. Compile for Coral Edge TPU
# ============================================================================

print("="*80)
print("CORAL EDGE TPU COMPILATION")
print("="*80)

print("Compiling model for Google Coral Edge TPU...")
os.chdir(FINALOUTPUTFOLDER)
compile_result = os.system('edgetpu_compiler limelight_neural_detector_8bit.tflite')

if compile_result == 0:
    # Rename the compiled model
    os.system('mv limelight_neural_detector_8bit_edgetpu.tflite limelight_neural_detector_coral.tflite')
    # Clean up log file
    if os.path.exists('limelight_neural_detector_8bit_edgetpu.log'):
        os.remove('limelight_neural_detector_8bit_edgetpu.log')
    print("✓ Coral compilation successful!")
else:
    print("⚠ Warning: Coral compilation may have encountered issues")
    print("  Check the edgetpu_compiler output above for details")

os.chdir(HOMEFOLDER)
print()

# ============================================================================
# 10. Package Output
# ============================================================================

print("="*80)
print("PACKAGING OUTPUT")
print("="*80)

# Create zip file
zip_path = f'{HOMEFOLDER}limelight_detectors.zip'
if os.path.exists(zip_path):
    os.remove(zip_path)

print("Creating zip archive...")
os.system(f'cd {HOMEFOLDER} && zip -r limelight_detectors.zip final_output/')
print(f"✓ Package created: {zip_path}")
print()

# ============================================================================
# COMPLETE!
# ============================================================================

print("="*80)
print("TRAINING PIPELINE COMPLETE!")
print("="*80)
print()
print(f"Output directory: {FINALOUTPUTFOLDER}")
print(f"Package file: {zip_path}")
print()
print("Files created:")
print("  ✓ limelight_neural_detector_32bit.tflite    (32-bit float model)")
print("  ✓ limelight_neural_detector_8bit.tflite     (8-bit quantized model)")
print("  ✓ limelight_neural_detector_coral.tflite    (Coral Edge TPU compiled) ← Deploy this!")
print("  ✓ limelight_neural_detector_labels.txt      (class labels)")
print("  ✓ pipeline_file.config                      (training configuration)")
print()
print("To retrieve your models:")
print(f"  Files are in: {FINALOUTPUTFOLDER}")
print("  If you mounted a volume, they're already on your host machine")
print("  Or copy the zip:")
print(f"    docker cp <container_id>:{zip_path} ./")
print()
print("="*80)
print("Upload limelight_neural_detector_coral.tflite and")
print("limelight_neural_detector_labels.txt to your Limelight device!")
print("="*80)