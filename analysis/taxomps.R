library(tidyverse)
library(fs)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "llama-3.1-8b", "Text Only",
  "Llama-3.2-11B-Vision", "llama-3.1-8b", "Vision + Text",
  "Molmo-7B-D-0924", "qwen2-7b-molmo","Vision + Text",
  "Qwen2-7B", "qwen2-7b-molmo", "Text Only",
  "Qwen2-7B", "qwen2-7b-llava-ov", "Text Only",
  "llava-1.5-7b-hf", "vicuna-7b","Vision + Text",
  "vicuna-7b-v1.5", "vicuna-7b", "Text Only",
  "llava-onevision-qwen2-7b-ov-hf", "qwen2-7b-llava-ov", "Vision + Text",
  "Llama-3.2-11B-Vision-Instruct", "llama-3.1.8b-instruct", "Vision + Text",
  "Llama-3.1-8B-Instruct", "llama-3.1.8b-instruct", "Text Only",
  "llava-v1.6-mistral-7b-hf", "mistral-7b", "Vision + Text",
  "Mistral-7B-Instruct-v0.2", "mistral-7b", "Text Only",
  "Qwen2.5-VL-7B-Instruct", "qwen-2.5-7b-instruct", "Vision + Text",
  "Qwen2.5-7B-Instruct", "qwen-2.5-7b-instruct", "Text Only",
  "SmolLM2-135M", "smollm2-135m", "Text Only",
  "SmolLM2-360M", "smollm2-360m", "Text Only",
  "SmolLM2-1.7B", "smollm2-1.7b", "Text Only",
  "SmolVLM-256M-Base", "smollm2-135m", "Vision + Text",
  "SmolVLM-500M-Base", "smollm2-360m", "Vision + Text",
  "SmolVLM-Base", "smollm2-1.7b", "Vision + Text",
)

ns_minimal_pairs <- read_csv("data/things-taxonomic-sensitivity/taxomps-ns-qa.csv") %>%
  janitor::clean_names()

swapped_minimal_pairs <- read_csv("data/things-taxonomic-sensitivity/taxomps-swapped-qa.csv") %>%
  janitor::clean_names()

ns_results <- dir_ls("data/results/taxomps-ns-all-qa/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/taxomps-ns-all-qa/"),
    model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
  ) %>%
  select(-file) %>%
  inner_join(ns_minimal_pairs)

ns_results %>%
  select(model, item, ns_id, hypernym_pred, negative_pred) %>%
  mutate(ns_score = 0.125 * as.numeric(negative_pred == "No")) %>%
  select(-negative_pred) %>%
  pivot_wider(names_from = ns_id, values_from = ns_score) %>%
  mutate(
    hypernym_score = 0.5 * as.numeric(hypernym_pred == "Yes"),
    score = hypernym_score + ns_1 + ns_2 + ns_3 + ns_4
  ) %>%
  group_by(model) %>%
  summarize(
    acc = mean(score)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc)


correctness <- ns_results %>%
  group_by(model, item) %>%
  summarize(
    num_correct = sum(hypernym_pred == "Yes" & negative_pred == "No")
  ) %>%
  ungroup() 

correctness %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 4)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc)

swapped_results <- dir_ls("data/results/taxomps-swapped-qa/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/taxomps-swapped-qa/"),
    model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
  ) %>%
  select(-file) %>%
  inner_join(swapped_minimal_pairs)

swapped_results %>%
  select(model, item, hypernym_pred, swapped_pred) %>%
  mutate(
    hypernym_score = 0.5 * as.numeric(hypernym_pred == "Yes"),
    swapped_score = 0.5 * as.numeric(swapped_pred == "No"),
    score = hypernym_score + swapped_score
  ) %>%
  group_by(model) %>%
  summarize(
    acc = mean(score)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc)

correctness <- swapped_results %>%
  group_by(model, item) %>%
  summarize(
    num_correct = sum(hypernym_pred == "Yes" & swapped_pred == "No")
  ) %>%
  ungroup() 

correctness %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 1)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc)

