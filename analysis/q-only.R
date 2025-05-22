library(tidyverse)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama_3.1_8B", "llama-3.1-8b", "Text Only",
  "mllama", "llama-3.1-8b", "Vision + Text",
  "molmo_D", "qwen2-7b-molmo","Vision + Text",
  "Qwen2_7B", "qwen2-7b-molmo", "Text Only",
  "Qwen2_7B_Instruct", "qwen2-7b-llava-ov", "Text Only",
  "llava", "vicuna-7b","Vision + Text",
  "vicuna_7b_v1.5", "vicuna-7b", "Text Only",
  "llava_ov", "qwen2-7b-llava-ov", "Vision + Text",
  "mllama_instruct", "llama-3.1.8b-instruct", "Vision + Text",
  "Llama_3.1_8B_Instruct", "llama-3.1.8b-instruct", "Text Only",
  "llava_next", "mistral-7b", "Vision + Text",
  "Mistral_7B_Instruct_v0.2", "mistral-7b", "Text Only",
  "qwen2.5VL", "qwen-2.5-7b-instruct", "Vision + Text",
  "Qwen2.5_7B_Instruct", "qwen-2.5-7b-instruct", "Text Only",
  "SmolLM2-135M", "smollm2-135m", "Text Only",
  "SmolLM2-360M", "smollm2-360m", "Text Only",
  "SmolLM2-1.7B", "smollm2-1.7b", "Text Only",
  "SmolVLM-256M-Base", "smollm2-135m", "Vision + Text",
  "SmolVLM-500M-Base", "smollm2-360m", "Vision + Text",
  "SmolVLM-Base", "smollm2-1.7b", "Vision + Text",
)

# '''
# Llama-3.1/MLlama-3.2
# Llama-3.1-I/MLlama-3.2-I
# Vicuna/Llava-1.5
# Mistral-v0.2-I/Llava-Next
# Qwen2/Molmo-D
# Qwen2-I/Llava-OneVision
# Qwen2.5-I/Qwen2.5-VL-I
# '''

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

another_model_meta <- tribble(
  ~setting, ~class, ~name,
  "lm_", "llama-3.1-8b", "Llama-3.1",
  "vlm_text_", "llama-3.1-8b", "MLlama-3.2",
  "lm_", "llama-3.1.8b-instruct", "Llama-3.1-I",
  "vlm_text_", "llama-3.1.8b-instruct", "MLlama-3.2-I",
  "lm_", "vicuna-7b", "Vicuna",
  "vlm_text_", "vicuna-7b", "Llava-1.5",
  "lm_", "mistral-7b", "Mistral-v0.2-I",
  "vlm_text_", "mistral-7b", "Llava-Next",
  "lm_","qwen2-7b-molmo", "Qwen2",
  "vlm_text_","qwen2-7b-molmo", "Molmo-D",
  "lm_","qwen2-7b-llava-ov", "Qwen2-I",
  "vlm_text_","qwen2-7b-llava-ov", "Llava-OV",
  "lm_","qwen-2.5-7b-instruct", "Qwen2.5-I",
  "vlm_text_","qwen-2.5-7b-instruct", "Qwen2.5-VL-I"
)

# results_raw <- read_csv("~/Downloads/updated_output_merged_strict_eval.csv") 
# results_raw <- read_tsv("~/Downloads/merged_model_results.csv")
results_raw <- read_csv("~/Downloads/final_model_outputs_9_types.csv")

valid_types <- results_raw %>% 
  filter(substitution_hop <0) %>% 
  count(question_type) %>%
  pull(question_type)

longer <- results_raw %>%
  select(-question, -input, -ground_truth) %>%
  pivot_longer(lm_Llama_3.1_8B:vlm_text_qwen2.5VL, names_to = "model_setting", values_to = "outcome") %>%
  mutate(
    is_ns = case_when(
      substitution_hop < 0 ~ TRUE,
      TRUE ~ FALSE
    ),
    substitution_hop = case_when(
      # substitution_hop >= 0 ~ substitution_hop,
      substitution_hop == -100 ~ 0,
      substitution_hop == -1 ~ 1,
      substitution_hop == -2 ~ 2,
      substitution_hop == -3 ~ 3,
      substitution_hop == -4 ~ 4,
      substitution_hop == -5 ~ 5,
      TRUE ~ substitution_hop
    )
  ) %>%
  filter(question_type %in% valid_types)


longer %>%
  filter(substitution_hop > 0, is_ns==FALSE) %>% count(argument)

longer %>%
  filter(substitution_hop > 0, is_ns==FALSE) %>%
  filter(str_detect(model_setting, "(lm_q_only|vlm_q_only)")) %>% 
  group_by(model_setting) %>%
  summarize(
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  )
