library(tidyverse)
library(lmerTest)

token_analysis_data_subset <- fs::dir_ls("data/token-analysis", regexp = "*.csv") %>%
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
  )

remove <- token_analysis_data_subset %>%
  filter(model == "Qwen2.5-VL-I") %>%
  count(question_id) %>%
  anti_join(
    token_analysis_data %>%
      filter(model == "Qwen2.5-I") %>%
      count(question_id)
  ) %>% pull(question_id)

token_analysis_data_final_subset <- token_analysis_data_subset %>%
  filter(!question_id %in% remove)

qwen_ids <- token_analysis_data_final_subset %>% distinct(question_id) %>% pull(question_id)

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
  token_analysis_data_subsample %>%
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
  # group_by(model, layer, subset) %>%
  # summarize(
  # acc = mean(correct == TRUE & neg_correct == TRUE)
  # )
  mutate(
    outcome = case_when(
      correct == TRUE & neg_correct == TRUE ~ "correct",
      TRUE ~ "incorrect"
    ),
    outcome = factor(outcome, levels = c("correct", "incorrect"), labels = c(1,0))
  ) %>%
  rename(pos_sim = sim) %>%
  mutate(
    diff = pos_sim - neg_sim,
    model = factor(model, levels = c("Qwen2.5-VL-I", "Qwen2.5-I"), labels = c("vlm", "lm"))
  ) 
# %>%
  # filter(question_id %in% qwen_ids)

demo <- reg_data %>%
  filter(layer == 23, model == "vlm")

fit <- glm(outcome ~ diff, data = demo, family = "binomial")
summary(fit)

broom::tidy(fit)

effectsize::standardize_parameters(fit) %>% as_tibble()


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
  ggplot(aes(layer, Std_Odds_Ratio, color = model, fill = model)) +
  geom_point() +
  geom_line() +
  geom_ribbon(aes(ymin = CI_low, ymax = CI_high), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed") 
