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
# results_raw <- read_csv("~/Downloads/final_model_outputs_9_types.csv")
results_raw <- read_csv("~/Downloads/model_inference_output.tsv")

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


### overall - pos only: vqa vs scene

ns_results <- longer %>%
  filter(is_ns == TRUE) %>%
  group_by(question_id, model_setting, substitution_hop) %>%
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
  group_by(question_id, model_setting, substitution_hop) %>%
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

  


overall_results <- with_ns %>%
  group_by(model_setting) %>%
  summarize(
    outcome = mean(correct == TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  inner_join(model_meta) %>%
  select(class, type, outcome) %>%
  pivot_wider(names_from = type, values_from = outcome) %>%
  mutate(
    diff = `Vision + Text` - `Text Only`
  ) %>%
  select(-diff)

with_ns %>%
  filter(substitution_hop == 0) %>%
  group_by(model_setting) %>%
  summarize(
    base = mean(correct)
  )

conditional <- with_ns %>%
  filter(substitution_hop == 0) %>%
  filter(correct == TRUE) %>%
  select(question_id, model_setting) %>%
  mutate(
    og_correct = TRUE
  ) %>%
  inner_join(with_ns %>% filter(substitution_hop != 0))

# conditional_new <- with_ns %>%
#   filter(substitution_hop == 0) %>%
#   select(question_id, model_setting, og_correct = correct) %>%
#   inner_join(with_ns) %>%
#   mutate(
#     cond_select = case_when(
#       og_correct == FALSE & substitution_hop > 0 ~ FALSE,
#       TRUE ~ TRUE
#     )
#   ) %>%
#   filter(cond_select == TRUE)
#   
# 
# # with_ns %>%
# #   filter(substitution_hop == 0) %>%
# #   filter(correct == TRUE) %>% count(question_id, model_setting, sort=TRUE)

conditional_results <- conditional %>%
  group_by(model_setting) %>%
  summarize(
    outcome = mean(correct==TRUE)
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

# conditional_new_results <- conditional_new %>%
#   group_by(model_setting) %>%
#   summarize(
#     outcome = mean(correct==TRUE)
#   ) %>%
#   ungroup() %>%
#   mutate(
#     setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
#     model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
#   ) %>%
#   filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
#   inner_join(model_meta) %>%
#   select(class, type, outcome) %>%
#   pivot_wider(names_from = type, values_from = outcome, values_fill = 0)


hca_results <- with_ns %>%
  group_by(question_id, model_setting) %>% 
  summarize(
    outcome = sum(correct == TRUE),
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
  overall_results %>% mutate(metric = "Overall"),
  conditional_results %>% mutate(metric = "Conditional"),
  # conditional_new_results %>% mutate(metric = "Conditional (New)")
) %>%
  mutate(
    metric = factor(metric, levels = c("Overall", "Conditional (New)", "Conditional", "HCA"))
  ) %>%
  inner_join(real_model_meta) %>%
  ggplot(aes(`Text Only`, `Vision + Text`, color = pair, shape = pair, fill = pair)) +
  geom_point(size = 3) +
  geom_abline(slope = 1, linetype = "dashed", linewidth = 0.2) +
  facet_wrap(~metric, nrow = 1) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 8, 9)) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_x_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "LM", y = "VLM"
  )

ggsave("plots/gqa-results.pdf", width = 10.44, height = 4.81, dpi = 300, device=cairo_pdf)
# ggsave("plots/gqa-results.svg", width = 10.44, height = 4.81, dpi = 300)
ggsave("plots/gqa-results-diffres.svg", width = 8.97, height = 4.26, dpi = 300)


ggsave("plots/gqa-results-alt-legend.pdf", width = 11.78, height = 3.35, dpi = 300, device = cairo_pdf)



