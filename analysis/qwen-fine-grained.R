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

# results_raw <- read_csv("~/Downloads/updated_output_merged_strict_eval.csv") 
# results_raw <- read_tsv("~/Downloads/merged_model_results.csv")
# results_raw <- read_csv("~/Downloads/final_model_outputs_9_types.csv")
results_raw <- read_csv("~/Downloads/model_inference_output.tsv")

valid_types <- results_raw %>% 
  filter(substitution_hop <0) %>% 
  count(question_type) %>%
  pull(question_type)

results_raw %>% 
  filter(substitution_hop >= 0) %>%
  filter(question_type %in% valid_types) %>%
  count(question_type, ground_truth)

valid_no <- c("existAttrC", "existAttrNotC", "existMaterialC", "existMaterialNotC", "existThatC", "existThatNotC")

longer <- results_raw %>%
  select(question_id, question_type, substitution_hop, original_arg, ground_truth, lm = lm_Qwen2.5_7B_Instruct, vlm = vlm_text_qwen2.5VL) %>%
  pivot_longer(lm:vlm, names_to = "model", values_to = "outcome") %>%
  mutate(
    is_ns = case_when(
      substitution_hop < 0 ~ TRUE,
      TRUE ~ FALSE
    ),
    substitution_hop = case_when(
      # substitution_hop >= 0 ~ substitution_hop,
      substitution_hop == -100 ~ 0,
      substitution_hop < 0 ~ -substitution_hop,
      TRUE ~ substitution_hop
    )
  ) %>%
  filter(question_type %in% valid_no)


ns_results <- longer %>%
  filter(is_ns == TRUE) %>%
  group_by(question_id, original_arg, model, substitution_hop) %>%
  summarize(
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>% 
  mutate(
    outcome = case_when(
      outcome != 1 ~ 0,
      outcome == 1 ~ 1
    )
  )

pos_results <- longer %>%
  filter(is_ns == FALSE) %>%
  group_by(question_id, original_arg, model, substitution_hop) %>%
  summarize(
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    outcome = case_when(
      outcome != 1 ~ 0,
      outcome == 1 ~ 1
    )
  )

with_ns <- ns_results %>%
  rename(neg_outcome = outcome) %>%
  inner_join(pos_results %>% rename(pos_outcome = outcome)) %>%
  mutate(correct = (neg_outcome == 1 & pos_outcome == 1))

with_ns %>% count(model, correct)

# with_ns %>%
#   filter(substitution_hop == 0) %>%
#   filter(correct == TRUE) %>%
#   select(model, question_id) %>%
#   write_csv("data/gqa_dataset/qwen-base-correct.csv")

with_ns %>%
  group_by(model) %>%
  summarize(
    acc = mean(correct)
  )

ns_details <- results_raw %>%
  select(question_id, substitution_hop, argument, original_arg) %>%
  filter(substitution_hop < 0) %>%
  mutate(
    substitution_hop = case_when(
      substitution_hop == -100 ~ 0,
      TRUE ~ -substitution_hop
    )
  ) %>%
  group_by(question_id, substitution_hop) %>%
  mutate(
    ns_id = glue::glue("neg{row_number()}")
  ) %>%
  ungroup() %>%
  pivot_wider(names_from = ns_id, values_from = argument)

pos_details <- results_raw %>%
  select(question_id, question_type, substitution_hop, argument, original_arg) %>%
  filter(substitution_hop >= 0)


all_data <- with_ns %>% 
  # filter(model_setting %in% c("lm_Qwen2.5_7B_Instruct", "vlm_text_qwen2.5VL")) %>%
  inner_join(ns_details) %>% inner_join(pos_details) %>%
  mutate(
    # # model = case_when(
    # #   str_detect(model_setting, "VL") ~ "Qwen2.5-VL-I",
    # #   TRUE ~ "Qwen2.5-I"
    # # )
    # model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  inner_join(another_model_meta) %>%
  select(question_id, question_type, orig_target = original_arg, hypernym = argument, neg1, neg2, neg3, neg4, model=name, correct) %>%
  mutate(
    correct = as.numeric(correct)
  ) 

# all_data %>%
#   mutate(
#     hypernym = case_when(
#       hypernym == "sports equiment" ~ "sports equipment",
#       TRUE ~ hypernym
#     ),
#     neg1 = case_when(
#       neg1 == "sports equiment" ~ "sports equipment",
#       TRUE ~ neg1
#     ),
#     neg2 = case_when(
#       neg2 == "sports equiment" ~ "sports equipment",
#       TRUE ~ neg2
#     ),
#     neg3 = case_when(
#       neg3 == "sports equiment" ~ "sports equipment",
#       TRUE ~ neg3
#     ),
#     neg4 = case_when(
#       neg4 == "sports equiment" ~ "sports equipment",
#       TRUE ~ neg4
#     )
#   ) %>% write_csv("data/all_negative_sampling_data.csv")


# subset for qwen

no_types <- results_raw %>% 
  filter(question_type %in% valid_types) %>% 
  count(question_type, ground_truth) %>%
  pivot_wider(names_from = ground_truth, values_from = n, values_fill = 0) %>%
  filter(yes == 0) %>%
  pull(question_type)

all_data %>%
  filter(model %in% c("Qwen2.5-I", "Qwen2.5-VL-I")) %>% count(model, correct)

qwen_correctness <- with_ns %>% 
  # filter(model_setting %in% c("lm_Qwen2.5_7B_Instruct", "vlm_text_qwen2.5VL")) %>%
  inner_join(ns_details) %>% inner_join(pos_details) %>%
  select(question_id, question_type, orig_target = original_arg, hypernym = argument, neg1, neg2, neg3, neg4, model, correct) %>%
  mutate(
    correct = as.numeric(correct)
  ) %>%
  filter(orig_target == hypernym) %>% filter(correct == 1) %>%
  filter(question_type %in% no_types)

qwen_correctness %>%
  filter(model == "vlm") %>%
  write_csv("data/gqa_dataset/qwen-vl-base-correct-no.csv")

qwen_correctness %>%
  filter(model == "lm") %>%
  write_csv("data/gqa_dataset/qwen-lm-base-correct-no.csv")


