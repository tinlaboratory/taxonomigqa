library(tidyverse)
library(lmerTest)

# token_analysis_data_subset <- fs::dir_ls("data/token-analysis", regexp = "*.csv") %>%
#   map_df(read_csv, .id = "model") %>%
#   mutate(
#     model = case_when(
#       str_detect(model, "VL") ~ "Qwen2.5-VL-I",
#       TRUE ~ "Qwen2.5-I"
#     ),
#     correct = case_when(
#       str_detect(model, "VL") ~ vlm_text_qwen2.5VL,
#       TRUE ~ lm_Qwen2.5_7B_Instruct
#     )
#   )
# 
# remove <- token_analysis_data_subset %>%
#   filter(model == "Qwen2.5-VL-I") %>%
#   count(question_id) %>%
#   anti_join(
#     token_analysis_data %>%
#       filter(model == "Qwen2.5-I") %>%
#       count(question_id)
#   ) %>% pull(question_id)
# 
# token_analysis_data_final_subset <- token_analysis_data_subset %>%
#   filter(!question_id %in% remove)
# 
# qwen_ids <- token_analysis_data_final_subset %>% distinct(question_id) %>% pull(question_id)

token_analysis_data <- fs::dir_ls("data/token-analysis-all/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = case_when(
      str_detect(model, "VL") ~ "Qwen2.5-VL-I",
      TRUE ~ "Qwen2.5-I"
    ),
    correct = case_when(
      str_detect(model, "VL") ~ vlm_text_qwen2.5VL,
      TRUE ~ lm_Qwen2.5_7B_Instruct
    )
  ) %>%
  group_by(model) %>%
  mutate(
    idx = row_number()-1
  ) %>%
  ungroup()


sims <- fs::dir_ls("data/results/gqa-cwe-sims-all/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model", col_names = c("idx", "layer", "sim")) %>%
  mutate(
    model = case_when(
      str_detect(model, "VL") ~ "Qwen2.5-VL-I",
      TRUE ~ "Qwen2.5-I"
    )
  )

negative_samples <- token_analysis_data %>%
  mutate(
    is_ns = substitution_hop < 0,
    substitution_hop = abs(substitution_hop)
  ) %>%
  filter(is_ns == TRUE) %>%
  group_by(model, question_id, substitution_hop, original_arg) %>%
  summarize(
    neg_correct = mean(correct)
  ) %>%
  ungroup()

positive_samples <- token_analysis_data %>%
  mutate(
    is_ns = substitution_hop < 0,
    substitution_hop = abs(substitution_hop)
  ) %>%
  filter(is_ns == FALSE) %>%
  group_by(model, question_id, substitution_hop, original_arg) %>%
  summarize(
    pos_correct = mean(correct)
  ) %>%
  ungroup()

obj_accs <- negative_samples %>%
  inner_join(positive_samples) %>%
  group_by(model, original_arg) %>%
  summarize(
    n = n(),
    accuracy = mean(neg_correct==1 & pos_correct ==1)
  ) %>% pivot_wider(names_from = model, values_from = c(accuracy, n)) %>%
  mutate(
    # diff = `Qwen2.5-VL-I` - `Qwen2.5-I`
    diff = `accuracy_Qwen2.5-VL-I` - `accuracy_Qwen2.5-I`,
    n = `n_Qwen2.5-I`
  ) 

candidates <- obj_accs %>%
  filter(n >= 10) %>%
  pull(original_arg)

set.seed(1024)
token_analysis_data_sample <- token_analysis_data %>%
  filter(substitution_hop > 0) %>%
  filter(model == "Qwen2.5-VL-I") %>%
  filter(original_arg %in% candidates) %>%
  group_by(original_arg) %>%
  sample_n(10)

sampled_question_ids <- token_analysis_data_sample %>% 
  distinct(question_id) %>%
  pull(question_id)

token_analysis_data_subsample <- token_analysis_data %>%
  filter(question_id %in% sampled_question_ids)

full_data <- sims %>% inner_join(
  token_analysis_data %>%
    select(model, idx, question_id, substitution_hop, correct)
) %>%
  group_by(model, layer) %>%
  mutate(
    new_idx = row_number()
  ) %>%
  ungroup() %>%
  mutate(
    is_ns = substitution_hop < 0,
    substitution_hop = abs(substitution_hop)
  )

ns <- full_data %>% filter(is_ns == TRUE) %>%
  group_by(model, layer, question_id, substitution_hop) %>%
  summarize(
    neg_correct = (mean(correct) == 1),
    neg_sim = mean(sim)
  ) %>%
  ungroup()

full_data %>% filter(is_ns != TRUE) %>% 
  inner_join(ns) %>%
  rename(pos_sim = sim) %>%
  filter(layer == 0) %>%
  group_by(model, layer) %>%
  summarize(
    acc = mean(correct),
    # sim = mean(pos_sim - neg_sim)
  ) %>%
  pivot_wider(names_from = model, values_from = sim) %>% View()



reg_data <- full_data %>% filter(is_ns != TRUE) %>% 
  inner_join(ns) %>%
  mutate(
    outcome = case_when(
      correct == TRUE & neg_correct == TRUE ~ 1,
      TRUE ~ 0
    )
    # outcome = as.integer(correct == TRUE & neg_correct == TRUE)
  ) %>%
  rename(pos_sim = sim) %>%
  mutate(
    diff = pos_sim - neg_sim,
    model = factor(model, levels = c("Qwen2.5-VL-I", "Qwen2.5-I"), labels = c("vlm", "lm")),
    # outcome = case_when()
    # outcome = factor(outcome, levels = c("incorrect", "correct"), labels = c(0, 1))
  ) 
# %>%
  # filter(question_id %in% qwen_ids)

demo <- reg_data %>%
  filter(layer == 10, model == "vlm")

reg_data

fit <- glm(outcome ~ diff, data = demo, family = binomial)
summary(fit)

broom::tidy(fit)

effectsize::standardize_parameters(fit) %>% as_tibble()

# set.seed(123)
# df <- data.frame(
#   predictor = rnorm(100, mean = 50, sd = 10),
#   outcome = rbinom(100, 1, 0.5)  # Binary outcome (0 or 1)
# )
# 
# # Fit logistic regression
# model <- glm(factor(outcome, levels = c(1, 0)) ~ predictor, data = df, family = binomial)
# 
# # See model summary
# summary(model)

reg_data %>%
  group_by(model, layer, outcome) %>%
  summarize(
    n = n(),
    sd = sd(diff),
    ste = qt(1 - (0.05/2), n - 1) * sd/sqrt(n),
    diff = mean(diff)
  ) %>% ggplot(aes(layer, diff, color = outcome, fill = outcome)) +
  geom_line() +
  geom_ribbon(aes(ymin = diff - ste, ymax = diff + ste), color = NA, alpha = 0.3) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  facet_wrap(~ model)


logregged <- reg_data %>%
  group_by(model, layer) %>%
  nest() %>%
  mutate(
    fit = map(data, function(x) {
      fit <- glm(outcome ~ diff, data = x, family = "binomial")
      p_val = broom::tidy(fit) %>% filter(term == "diff") %>% pull(p.value)
      
      eff_size <- effectsize::standardize_parameters(fit, exp=TRUE) %>% 
        as_tibble() %>%
        filter(Parameter == "diff") %>%
        mutate(p_val = p_val)
      
      return(eff_size)
    })
  ) %>%
  select(-data) %>%
  unnest(cols = fit)

logregged %>%
  mutate(
    model = str_to_upper(model),
    model = factor(model, levels = c("VLM", "LM"))
  ) %>%
  ggplot(aes(layer, Std_Odds_Ratio, color = model, fill = model, linetype = model)) +
  geom_line(linewidth = 0.8) +
  geom_ribbon(aes(ymin = CI_low, ymax = CI_high), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 1, linetype = "dotted") +
  scale_y_continuous(breaks = scales::pretty_breaks(6)) +
  scale_x_continuous(breaks = scales::pretty_breaks(6)) +
  scale_color_manual(values = c("#bf812d", "#35978f"), aesthetics = c("color", "fill")) +
  # scale_color_manual(values = c("#d8b365", "#5ab4ac"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Layer (0-28)",
    y = "Odds Ratio (>1 = more correct)",
    color = "Model",
    fill = "Model",
    linetype = "Model"
  )

ggsave("plots/token-sim-log-reg-qwen2.5-lm-vlm.pdf", height = 4.61, width = 4.20, dpi = 300, device=cairo_pdf)
