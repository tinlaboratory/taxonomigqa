library(tidyverse)
library(lmerTest)

token_analysis_data <- fs::dir_ls("data/token-analysis-all", regexp = "*.csv") %>%
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

remove <- token_analysis_data %>%
  filter(model == "Qwen2.5-VL-I") %>%
  count(question_id) %>%
  anti_join(
    token_analysis_data %>%
      filter(model == "Qwen2.5-I") %>%
      count(question_id)
  ) %>% pull(question_id)

token_analysis_data_final <- token_analysis_data %>%
  filter(!question_id %in% remove)

qwen_ids <- token_analysis_data_final %>% distinct(question_id) %>% pull(question_ids)

sims <- fs::dir_ls("data/results/gqa-cwe-sims/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model", col_names = c("idx", "layer", "sim")) %>%
  mutate(
    model = case_when(
      str_detect(model, "VL") ~ "Qwen2.5-VL-I",
      TRUE ~ "Qwen2.5-I"
    )
  )


token_analysis_data %>% count(model, subset)

sims %>% filter(layer == 1) %>% count(model)

full_data <- sims %>% inner_join(
  token_analysis_data_final %>%
    select(model, idx, question_id, subset, substitution_hop, correct)
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
    outcome = factor(outcome)
  ) %>%
  rename(pos_sim = sim) %>%
  mutate(
    diff = pos_sim - neg_sim,
    model = factor(model, levels = c("Qwen2.5-VL-I", "Qwen2.5-I"), labels = c("vlm", "lm"))
  )


reg_data


demo <- reg_data %>%
  filter(layer == 23, model == "lm")

fit <- glm(outcome ~ diff, data = demo, family = "binomial")
summary(fit)

broom::tidy(fit)

effectsize::standardize_parameters(fit) %>% as_tibble()


logregged <- reg_data %>%
  group_by(model, layer) %>%
  nest() %>%
  mutate(
    fit = map(data, function(x) {
      fit <- glm(outcome ~ diff + subset, data = x, family = "binomial")
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

logregged$p_val


logregged %>%
  ggplot(aes(layer, Std_Odds_Ratio, color = model, fill = model)) +
  geom_point() +
  geom_line() +
  geom_ribbon(aes(ymin = CI_low, ymax = CI_high), color = NA, alpha = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed") 


ll.null <- fit$null.deviance/-2
ll.proposed <- fit$deviance/-2

(ll.null - ll.proposed)/ll.null

1 - pchisq(2*(ll.proposed - ll.null), df = length(fit$coefficients)-1)

