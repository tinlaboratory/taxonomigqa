library(tidyverse)
library(fs)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "llama-3.1-8b", "Text Only",
  "Llama-3.2-11B-Vision", "llama-3.1-8b", "Vision + Text",
  "Molmo-7B-D-0924", "qwen-2-7b-molmo","Vision + Text",
  "Qwen2-7B", "qwen-2-7b-molmo", "Text Only",
  # "Qwen2-7B", "qwen-2-7b-llava-ov", "Text Only",
  "Qwen2-7B-Instruct", "qwen2-7b-llava-ov", "Text Only",
  "llava-1.5-7b-hf", "vicuna-7b","Vision + Text",
  "vicuna-7b-v1.5", "vicuna-7b", "Text Only",
  "llava-onevision-qwen2-7b-ov-hf", "qwen-2-7b-llava-ov", "Vision + Text",
  "Llama-3.2-11B-Vision-Instruct", "llama-3.1-8b-instruct", "Vision + Text",
  "Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct", "Text Only",
  "llava-v1.6-mistral-7b-hf", "mistral-7b", "Vision + Text",
  "Mistral-7B-Instruct-v0.2", "mistral-7b", "Text Only",
  "Qwen2.5-VL-7B-Instruct", "qwen-2.5-7b-instruct", "Vision + Text",
  "Qwen2.5-7B-Instruct", "qwen-2.5-7b-instruct", "Text Only"
)

rsa_results <- read_csv("data/results/pair-rsa.csv") %>%
  rename(class = model_class)

rsa_results %>%
  filter(!str_detect(metric, "mean")) %>%
  filter(!str_detect(metric, "_s_")) %>%
  filter(!str_detect(metric, "param_diff")) %>%
  # mutate(
  #   score = case_when(
  #     score > 0.1 ~ round(score, digits = 4),
  #     TRUE ~ score
  #   )
  # ) %>%
  filter(str_detect(class, "llava-ov"))


