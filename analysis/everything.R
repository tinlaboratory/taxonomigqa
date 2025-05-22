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

results_raw <- read_csv("~/Downloads/output_merged_strict_eval.csv")

longer <- results_raw %>%
  select(-question, -scene_description, -ground_truth, -ground_truth_long) %>%
  pivot_longer(lm_Llama_3.1_8B:vlm_text_qwen2.5VL, names_to = "model_setting", values_to = "outcome")

# vlm_*
# vlm_q_only_*
# vlm_text_*
# lm_*
# lm_q_only_*

# overall
overall <- longer %>%
  group_by(question_type, model_setting) %>%
  summarize(
    n = n(),
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(question_type, class, type, outcome, n) %>%
  pivot_wider(names_from = type, values_from = outcome) %>%
  mutate(
    diff = `Vision + Text` - `Text Only`
  )

overall_results <- longer %>%
  group_by(model_setting) %>%
  summarize(
    n = n(),
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(class, type, outcome, n) %>%
  pivot_wider(names_from = type, values_from = outcome) %>%
  mutate(
    diff = `Vision + Text` - `Text Only`
  ) %>%
  select(-n, -diff)

overall %>%
  mutate(question_type = glue::glue("{question_type} ({n})")) %>%
  group_by(class) %>%
  summarize(
    vision_wins = mean(diff > 0),
    text_wins = mean(diff < 0),
    tie = mean(diff == 0)
  )

overall %>%
  mutate(question_type = glue::glue("{question_type} ({n})")) %>%
  mutate(
    question_type = factor(question_type),
    question_type = fct_reorder(question_type, n)
  ) %>%
  ggplot(aes(`Text Only`, `Vision + Text`, color = class, shape = class, fill = class)) +
  geom_point(size = 2) +
  geom_abline(slope = 1, linetype = "dashed", linewidth = 0.2) +
  facet_wrap(~question_type) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 8, 9)) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  theme(legend.position = "top")
  # scale_color_manual(values = c(), aesthetics = c("color", "fill"))

# og_questions <- 
conditional <- longer %>%
  filter(substitution_hop == 0) %>%
  filter(outcome == TRUE) %>%
  select(question_id, model_setting) %>%
  mutate(
    og_correct = TRUE
  ) %>%
  inner_join(longer %>% filter(substitution_hop != 0))

conditional_results <- conditional %>%
  group_by(model_setting) %>%
  summarize(
    outcome = mean(outcome==TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(class, type, outcome) %>%
  pivot_wider(names_from = type, values_from = outcome, values_fill = 0)

conditional_typewise <- conditional %>%
  group_by(question_type, model_setting) %>%
  summarize(
    n = n(),
    outcome = mean(outcome == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(question_type, class, type, outcome) %>%
  pivot_wider(names_from = type, values_from = outcome, values_fill = 0) %>%
  mutate(
    diff = `Vision + Text` - `Text Only`
  )

conditional_typewise %>%
  group_by(class) %>%
  summarize(
    vision_wins = mean(diff > 0),
    text_wins = mean(diff < 0),
    tie = mean(diff == 0)
  )


hca_results <- longer %>%
  group_by(question_id, model_setting) %>% 
  summarize(
    outcome = sum(outcome == TRUE),
    n = n()
  ) %>%
  ungroup() %>%
  group_by(model_setting) %>%
  summarize(outcome = mean(outcome == n)) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(class, type, outcome) %>%
  pivot_wider(names_from = type, values_from = outcome, values_fill = 0)


bind_rows(
  hca_results %>% mutate(metric = "HCA"),
  overall_results %>% mutate(metric = "Conditional"),
  conditional_results %>% mutate(metric = "Overall")
) %>%
  mutate(
    metric = factor(metric, levels = c("Overall", "Conditional", "HCA"))
  ) %>%
  inner_join(real_model_meta) %>%
  ggplot(aes(`Text Only`, `Vision + Text`, color = pair, shape = pair, fill = pair)) +
  geom_point(size = 3) +
  geom_abline(slope = 1, linetype = "dashed", linewidth = 0.2) +
  facet_wrap(~metric) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 8, 9)) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_x_continuous(limits = c(0,1)) +
  scale_y_continuous(limits = c(0,1)) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 12)
  )
  
