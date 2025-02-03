# # from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# # from PIL import Image
# # import requests

# # model_id = "google/paligemma2-10b-ft-docci-448"
# # model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
# # model = model.to("cuda")
# # processor = AutoProcessor.from_pretrained(model_id)

# # prompt = "<image>caption en"
# # image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
# # raw_image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")

# # inputs = processor(prompt, raw_image, return_tensors="pt").to("cuda")
# # output = model.generate(**inputs, max_new_tokens=200)

# # input_len = inputs["input_ids"].shape[-1]
# # print(processor.decode(output[0][input_len:], skip_special_tokens=True))
# # # A medium shot of two cats laying on a pile of brown fishing nets. The cat in the foreground is a gray tabby cat with white on its chest and paws. The cat is laying on its side with its head facing the bottom right corner of the image. The cat in the background is laying on its side with its head facing the top left corner of the image. The cat's body is curled up, its head is slightly turned to the right, and its front paws are tucked underneath its body. There is a teal rope hanging from the fishing net in the top right corner of the image.


# from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
# from transformers.image_utils import load_image
# import torch

# model_id = "google/paligemma2-3b-pt-224"
# IMAGE_PATH = "/projectnb/tin-lab/yuluq/data/gqa/data/temp_image/2323902.jpg"
# IMAGE_PATH1 = "/projectnb/tin-lab/yuluq/data/gqa/data/temp_image/2350497.jpg"
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

# image1 = load_image(IMAGE_PATH)
# image2 = load_image(IMAGE_PATH1)
# images = [image1, image2]
# print("image1:", type(image1))
# image = load_image(url)
# print(type(image))
# model = PaliGemmaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.bfloat16, device_map="auto"
#     ).eval()

# processor = AutoProcessor.from_pretrained(model_id)

# # Leaving the prompt blank for pre-trained models
# prompt = ["what is in the image? ", "what is in the image, a train? "]
# tokenized_output = processor.tokenizer(prompt, return_tensors="pt")

# # print("special tokens: ", processor.tokenizer.special_tokens_map)
# # print("Token IDs:", tokenized_output['input_ids'])
# # print("Decoded Tokens:", [processor.tokenizer.decode(id) for id in tokenized_output['input_ids'][0]])
# model_inputs = processor(
#     text=prompt, 
#     images=images, 
#     return_tensors="pt", 
#     truncation=True,
#     padding=True
#     ).to(torch.bfloat16).to(model.device)
# print(type(model_inputs))
# print("token:", [processor.tokenizer.decode(id) for id in model_inputs["input_ids"][1]])
# exit()
# input_len = model_inputs["input_ids"].shape[-1]
# print(len(model_inputs))
# print(model_inputs)
# print(model_inputs['input_ids'].shape)
# print(model_inputs['pixel_values'].shape)
# exit()
# with torch.inference_mode():
#     generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
#     decoded = processor.decode(generation, skip_special_tokens=True)
#     print(decoded)

