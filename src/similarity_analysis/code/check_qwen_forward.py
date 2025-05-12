import torch
import requests
from PIL import Image
from io import BytesIO
import gc # Garbage collector for memory management verification

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BATCH_SIZE = 32 # As requested
# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Model & Processor ---
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model and processor loaded.")

# --- Prepare Batched Input ---
print(f"Creating {BATCH_SIZE} dummy images...")
# Create slightly varied dummy images for clarity if needed, but same size is fine
# dummy_images = [Image.new('RGB', (448, 448), color = (min(i*5, 255), 100, 50)) for i in range(BATCH_SIZE)]

# Paths
img_paths = ['/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_09s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_14n.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_07s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_10s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_11s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_01b.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_12s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_03s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_13s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_06s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_02s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_05s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_15s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_08s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/hairbrush/hairbrush_04s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_01b.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_13s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_04s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_11s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_06s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_09s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_14s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_10s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_05s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_03s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_07s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_12s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_02s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/nail_polish/nail_polish_08s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/deodorant/deodorant_08s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/deodorant/deodorant_04s.jpg', '/projectnb/tin-lab/yuluq/data/THINGS/object_images/deodorant/deodorant_11s.jpg']
dummy_images = [Image.open(path).convert('RGB').resize((448, 448)) for path in img_paths]

images_batch = dummy_images # This is your list of PIL Images

print("Preparing messages for chat template...")
# Create the message structure for each item in the batch
messages_list = []
for i in range(BATCH_SIZE):
    # Each element in the list passed to apply_chat_template should be a conversation
    messages_list.append(
        [
            {"role": "user", "content": [
                # IMPORTANT: Use a placeholder dict for the image here.
                # The processor links it to the actual image passed later.
                {"type": "image"},
                {"type": "text", "text": f"Describe image {i+1}."} # Slightly different prompts
            ]}
        ]
    )

print("Applying chat template...")
# Apply the template to each conversation in the batch
# This generates the text prompts with image placeholders correctly inserted
# Returns a list of formatted strings
text_inputs_templated = processor.apply_chat_template(
    messages_list,
    tokenize=False, # Get the raw text string first
    add_generation_prompt=True # Important for inference format
)

print("Processing inputs with text and images...")
# Now, call the processor with the templated text and the actual images
# The processor will tokenize the text (including placeholders) and process images
inputs = processor(
    text=text_inputs_templated, # List of templated texts
    images=images_batch,      # List of PIL images
    padding=True,             # Pad sequences to the same length in the batch
    return_tensors="pt",
)

# Move inputs to the main device
try:
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    print(f"Inputs moved to {DEVICE}")
except Exception as e:
    print(f"Warning: Could not move all inputs to {DEVICE}. Error: {e}")

# --- Hook Function Definition ---
hook_data = {}

def vision_tower_forward_hook(module, hook_input, hook_output):
    print("\n--- Vision Tower Forward Hook Triggered ---")
    module_device = next(module.parameters()).device
    current_mem = torch.cuda.memory_allocated(module_device) / (1024**2) # MiB
    peak_mem = torch.cuda.max_memory_allocated(module_device) / (1024**2) # MiB
    print(f"Device of Vision Tower: {module_device}")
    print(f"Memory Allocated on {module_device} (after vision tower fwd): {current_mem:.2f} MiB")
    print(f"Peak Memory Allocated on {module_device} (since reset/start): {peak_mem:.2f} MiB")
    hook_data['memory_after_vision_fwd_mib'] = current_mem
    hook_data['peak_memory_mib'] = peak_mem
    hook_data['vision_tower_device'] = module_device
    # Optionally check output shape
    if isinstance(hook_output, (list, tuple)) and len(hook_output) > 0 and isinstance(hook_output[0], torch.Tensor):
         print(f"Vision tower output features shape: {hook_output[0].shape}")
    elif isinstance(hook_output, torch.Tensor):
         print(f"Vision tower output features shape: {hook_output.shape}")


# --- Register Hook & Run Forward Pass ---
try:
    vision_tower = model.visual
    vision_tower_device = next(vision_tower.parameters()).device
    print(f"Vision tower located on device: {vision_tower_device}")
except AttributeError:
    print("Error: Could not find 'model.visual'. Check model architecture.")
    exit()

hook_handle = vision_tower.register_forward_hook(vision_tower_forward_hook)
print("Forward hook registered on vision tower.")

# --- Print Processor Outputs Before Forward ---
print("\n--- Information from Processor Output ---")
total_patch_count = 0
if 'image_grid_thw' in inputs:
    grid_thw = inputs['image_grid_thw'] # Shape should be [batch_size, 3]
    print(f"image_grid_thw shape: {grid_thw.shape}")
    print(f"image_grid_thw (first item): {grid_thw[0].tolist()}")
    print(f"image_grid_thw (full): {grid_thw}")
    # Calculate total patches across the batch
    patch_counts_per_image = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    total_patch_count = patch_counts_per_image.sum().item()
    print(f"Calculated total patch tokens across batch: {total_patch_count}")
    hook_data['image_grid_thw'] = grid_thw
    hook_data['total_patch_count'] = total_patch_count
else:
    print("'image_grid_thw' not found in processor output.")

# Print shapes of all tensor inputs
print(f"\nInput tensor shapes: {{k: v.shape for k, v in inputs.items() if isinstance(v, torch.Tensor)}}")
# Specifically check input_ids to see if it looks reasonable (should be > 0)
if 'input_ids' in inputs:
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"input_ids (first item sample): {inputs['input_ids'][0, :20]}...") # Print start of first sequence


# --- Perform Forward Pass ---
print("\n--- Performing Model Forward Pass (will trigger hook) ---")
if DEVICE == "cuda":
    gc.collect()
    torch.cuda.empty_cache()
    # Reset peak memory stats specifically on the vision tower's device
    if vision_tower_device.type == 'cuda':
         torch.cuda.reset_peak_memory_stats(vision_tower_device)
         mem_before = torch.cuda.memory_allocated(vision_tower_device) / (1024**2)
         print(f"Memory on {vision_tower_device} before forward pass: {mem_before:.2f} MiB")
    else:
         print("Vision tower not on CUDA, skipping CUDA memory stats.")

# Run the forward pass. We only need the outputs for the model to run, not for inspection here.
# Using `model.forward()` directly instead of `model.generate()`
with torch.no_grad():
    try:
        # Pass only the arguments the model's forward method expects
        # Typically 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw' etc.
        # **inputs passes all keys from the dict as keyword arguments.
        outputs = model(**inputs)
        print("Model forward pass completed.")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        # Clean up hook even if forward pass fails
        hook_handle.remove()
        print("Forward hook removed due to error.")
        exit()

if vision_tower_device.type == 'cuda':
    mem_after = torch.cuda.memory_allocated(vision_tower_device) / (1024**2)
    peak_mem_fwd = torch.cuda.max_memory_allocated(vision_tower_device) / (1024**2)
    print(f"Memory on {vision_tower_device} after forward pass: {mem_after:.2f} MiB")
    print(f"Peak memory on {vision_tower_device} during forward pass: {peak_mem_fwd:.2f} MiB")


# --- Cleanup ---
hook_handle.remove()
print("Forward hook removed.")

# --- Final Summary ---
print("\n--- Final Collected Data Summary ---")
if 'image_grid_thw' in hook_data:
    print(f"Captured image_grid_thw (shape): {hook_data['image_grid_thw'].shape}")
    print(f"Captured image_grid_thw (first item): {hook_data['image_grid_thw'][0].tolist()}")
    print(f"All of image_grid_thw: {hook_data['image_grid_thw']}")
if 'total_patch_count' in hook_data:
    print(f"Calculated total patch tokens: {hook_data['total_patch_count']}")
if 'memory_after_vision_fwd_mib' in hook_data:
    print(f"Memory on {hook_data.get('vision_tower_device', 'N/A')} reported by hook (after vision fwd): {hook_data['memory_after_vision_fwd_mib']:.2f} MiB")
if 'peak_memory_mib' in hook_data:
     print(f"Peak memory on {hook_data.get('vision_tower_device', 'N/A')} reported by hook: {hook_data['peak_memory_mib']:.2f} MiB")

print("\nScript finished.")