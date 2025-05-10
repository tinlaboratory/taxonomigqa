library(tidyverse)
library(fs)
library(glue)

hypernyms <- read_csv("data/gqa_entities/taxomps-hypernym.csv")
ns <- read_csv("data/gqa_entities/taxomps-ns-all.csv")
swapped <- read_csv("data/gqa_entities/taxomps-swapped.csv")

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "llama-3.1-8b", "Text Only",
  "Llama-3.2-11B-Vision", "llama-3.1-8b", "Vision + Text",
  "Molmo-7B-D-0924", "qwen2-7b-molmo","Vision + Text",
  "Qwen2-7B", "qwen2-7b-molmo", "Text Only",
  "Qwen2-7B-Instruct", "qwen2-7b-llava-ov", "Text Only",
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

real_model_meta <- tribble(
  ~class, ~pair,
  "llama-3.1-8b", "Llama-3.1 vs. MLlama-3.2",
  "llama-3.1.8b-instruct", "Llama-3.1-I vs. MLlama-3.2-I",
  "vicuna-7b", "Vicuna vs. Llava-1.5",
  "mistral-7b", "Mistral-v0.2-I vs. Llava-Next",
  "qwen2-7b-molmo", "Qwen2 vs. Molmo-D",
  "qwen2-7b-llava-ov", "Qwen2-I vs. Llava-OV",
  "qwen-2.5-7b-instruct", "Qwen2.5-I vs. Qwen2.5-VL-I"
)

model_meta %>% count(class)

read_taxomps_results <- function(subset="hypernym") {
  results <- dir_ls(glue("data/results/taxomps-{subset}-qa/"), regexp = "*.csv") %>%
    map_df(read_csv, .id = "file") %>%
    mutate(
      model = str_remove(file, glue("data/results/taxomps-{subset}-qa/")),
      model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
    ) %>%
    select(-file)
  
  return(results)
}

hypernym_results <- read_taxomps_results("hypernym") %>%
  inner_join(hypernyms) %>%
  group_by(model, category_id, parent_id) %>%
  slice_max(p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

swapped_results <- read_taxomps_results("swapped") %>%
  inner_join(swapped) %>%
  group_by(model, category_id, parent_id) %>%
  slice_max(-p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

ns_results <- read_taxomps_results("ns-all") %>%
  inner_join(ns) %>% 
  group_by(model, category_id, parent_id, ns_id) %>%
  slice_max(-p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

ns_experiment <- hypernym_results %>%
  select(model, category_id, parent_id, hypernym_label = label) %>%
  inner_join(ns_results %>% select(model, category_id, parent_id, ns_id, ns_label = label))

swapped_experiment <- hypernym_results %>%
  select(model, category_id, parent_id, hypernym_label = label) %>%
  inner_join(swapped_results %>% select(model, category_id, parent_id, swapped_label = label))


hypernym_results %>%
  group_by(model) %>%
  summarize(
    acc = mean(label == "Yes")
  )

#ns weighted 
ns_weighted <- ns_experiment %>%
  mutate(ns_score = 0.125 * as.numeric(ns_label == "No")) %>%
  select(-ns_label) %>%
  pivot_wider(names_from = ns_id, values_from = ns_score) %>%
  mutate(
    hypernym_score = 0.5 * as.numeric(hypernym_label == "Yes"),
    score = hypernym_score + ns_1 + ns_2 + ns_3 + ns_4
  ) %>%
  group_by(model) %>%
  summarize(
    acc = mean(score)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc) %>%
  mutate(
    vision_better = `Vision + Text` > `Text Only`
  ) %>%
  filter(!str_detect(class, "smollm"))

# strict scoring

ns_strict <- ns_experiment %>%
  group_by(model, category_id, parent_id) %>%
  summarize(
    num_correct = sum(hypernym_label == "Yes" & ns_label == "No")
  ) %>%
  ungroup() %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 4)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc) %>%
  mutate(
    vision_better = `Vision + Text` > `Text Only`
  ) %>%
  filter(!str_detect(class, "smollm"))


## swapped
# weighted 

swapped_weighted <- swapped_experiment %>%
  select(model, category_id, parent_id, hypernym_label, swapped_label) %>%
    mutate(
      hypernym_score = 0.5 * as.numeric(hypernym_label == "Yes"),
      swapped_score = 0.5 * as.numeric(swapped_label == "No"),
      score = hypernym_score + swapped_score
    ) %>%
    group_by(model) %>%
    summarize(
      acc = mean(score)
    ) %>%
    inner_join(model_meta, relationship = "many-to-many") %>%
    select(-model) %>%
    pivot_wider(names_from = type, values_from = acc) %>%
  mutate(
    vision_better = `Vision + Text` > `Text Only`
  ) %>%
  filter(!str_detect(class, "smollm"))


# strict
swapped_strict <- swapped_experiment %>%
  group_by(model, category_id, parent_id) %>%
  summarize(
    num_correct = sum(hypernym_label == "Yes" & swapped_label == "No")
  ) %>%
  ungroup() %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 1)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc) %>%
  mutate(
    vision_better = `Vision + Text` > `Text Only`
  ) %>%
  filter(!str_detect(class, "smollm"))


weighted <- ns_weighted %>%
  janitor::clean_names() %>%
  mutate(ns_diff = vision_text - text_only) %>%
  select(class, ns_diff) %>% 
  inner_join(swapped_weighted %>%
  janitor::clean_names() %>%
  mutate(swapped_diff = vision_text - text_only) %>%
  select(class, swapped_diff)) %>%
  mutate(metric = "weighted")


strict <- ns_strict %>%
  janitor::clean_names() %>%
  mutate(ns_diff = vision_text - text_only) %>%
  select(class, ns_text_only = text_only, ns_vision_text = vision_text, ns_diff) %>% 
  inner_join(swapped_strict %>%
               janitor::clean_names() %>%
               mutate(swapped_diff = vision_text - text_only) %>%
               select(class, swapped_text_only = text_only, swapped_vision_text = vision_text, swapped_diff)) %>%
  mutate(metric = "strict")

strict %>%
  mutate(
    agree = (ns_diff > 0 & swapped_diff > 0) | (ns_diff < 0 & swapped_diff < 0)
  ) %>%
  View()

weighted %>%
  mutate(
    agree = (ns_diff > 0 & swapped_diff > 0) | (ns_diff < 0 & swapped_diff < 0)
  )


strict %>%
  mutate(
    agree = (ns_diff > 0 & swapped_diff > 0) | (ns_diff < 0 & swapped_diff < 0)
  ) %>%
  select(-ns_diff, -metric, -agree, -swapped_diff) %>%
  inner_join(real_model_meta) %>%
  select(pair, -class, ns_text_only:swapped_vision_text)


