library(tidyverse)
library(lmerTest)


results_raw <- read_csv("~/Downloads/final_model_outputs_9_types.csv")

viz_sim <- read_tsv("~/Downloads/qwen_cosine_similarities.csv")

viz_sim

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

hypernyms <- longer %>%
  filter(is_ns == FALSE) %>%
  filter(model_setting == "lm_Qwen2.5_7B_Instruct") %>%
  distinct(question_id, substitution_hop, original_arg, argument)


### overall - pos only: vqa vs scene

ns_results <- longer %>%
  filter(is_ns == TRUE) %>%
  group_by(question_id, model_setting, substitution_hop, original_arg) %>%
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
  group_by(question_id, model_setting, substitution_hop, original_arg) %>%
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

conditional <- with_ns %>%
  filter(substitution_hop == 0) %>%
  filter(correct == TRUE) %>%
  select(question_id, model_setting) %>%
  mutate(
    og_correct = TRUE
  ) %>%
  inner_join(with_ns %>% filter(substitution_hop != 0))

qwen_diffs <- conditional %>%
  group_by(model_setting, substitution_hop, original_arg) %>%
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
  inner_join(hypernyms %>% distinct(original_arg, substitution_hop, argument)) %>%
  filter(model %in% c("Qwen2.5_7B_Instruct", "qwen2.5VL")) %>%
  select(setting, concept1 = original_arg, concept2 = argument, accuracy = outcome) %>%
  pivot_wider(names_from = setting, values_from = accuracy) %>%
  janitor::clean_names() %>%
  mutate(diff = vlm_text - lm)


qwen_diffs %>% 
  # add_count(category) %>%
  # filter(n >= 10) %>%
  anti_join(viz_sim)


joined <- viz_sim %>% 
  inner_join(qwen_diffs) %>%
  filter(!is.na(diff)) %>%
  select(-diff) %>%
  pivot_longer(lm:vlm_text, names_to = "type", values_to = "accuracy")

joined %>%
  add_count(category) %>%
  filter(n >= 27*2) %>%
  # pivot_longer(lm:vlm_text, names_to = "type", values_to = "accuracy") %>%
  ggplot(aes(similarity_Mean, accuracy, group = interaction(category, type))) +
  geom_point(aes(color = type)) +
  geom_smooth(aes(group = interaction(category, type), color = type), method = "lm") +
  facet_wrap(~category, scales = "free")


joined_reg <- joined %>%
  mutate(
    mean_sim = similarity_Mean,
    pairwise_sim = similarity_pairwise,
    category = factor(category),
    type = case_when(
      type == "lm" ~ -1,
      TRUE ~ 1
    )
  )

fit <- lmer(accuracy ~ mean_sim + (1 + mean_sim | concept2),REML = F, data = joined_reg %>% filter(type == 1))
fit_no_sim <- lmer(accuracy ~  (1 + mean_sim | concept2),REML = F, data = joined_reg %>% filter(type == 1))
fit_no_sim_all <- lmer(accuracy ~  (1 | concept2),REML = F, data = joined_reg %>% filter(type == 1))


anova(fit, fit_no_sim, fit_no_sim_all)

fit <- lmer(accuracy ~ mean_sim + (1 + mean_sim | concept2),REML = F, data = joined_reg %>% filter(type == 1))
fit_no_slope <- lmer(accuracy ~ mean_sim + (1  | concept2),REML = F, data = joined_reg %>% filter(type == 1))

anova(fit, fit_no_slope)
summary(fit)

ranef(fit)

coef(fit)

vlm_ranef <- rownames_to_column(ranef(fit)$concept2, var = "category") %>%
  # mutate(verb = factor(category, levels = fcts)) %>%
  mutate(category = factor(category), category = fct_reorder(category, mean_sim))

vlm_ranef %>%
  ggplot(aes(category, mean_sim)) +
  geom_col(fill = "#7570b3") +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 70, vjust = 0.6),
    axis.text = element_text(color = "black"),
    axis.title.x = element_blank(),
    plot.title = element_text(family="Inconsolata", face="bold")
  ) +
  labs(
    # y="Category-specific effect of Similarity\nrelative to global effect",
    y = "Relative Effect of Similarity"
    # title = "no datives"
  )
ggsave("plots/ranef-qwen-img-sim.pdf", dpi = 300, width = 7.77, height = 3.53, device=cairo_pdf)


fit_lm <- lmer(accuracy ~ mean_sim + (mean_sim | category),REML = F, data = joined_reg %>% filter(type == -1))
fit_lm2 <- lmer(accuracy ~ mean_sim + (mean_sim | category),REML = F, data = joined_reg %>% filter(type == -1))
summary(fit_lm)

ranef(fit_lm)

rownames_to_column(coef(fit_lm)$category, var = "category") %>%
  mutate(category = factor(category, levels = levels(vlm_ranef$category))) %>%
  # mutate(category = factor(category), category = fct_reorder(category, mean_sim)) %>%
  ggplot(aes(category, mean_sim)) +
  geom_col(fill = "#e6ab02") +
  theme_bw(base_size = 15, base_family = "Palatino") +
  theme(
    axis.text.x = element_text(angle = 70, vjust = 0.6),
    axis.text = element_text(color = "black"),
    axis.title.x = element_blank(),
    plot.title = element_text(family="Inconsolata", face="bold")
  ) +
  labs(
    y="Sim Effect",
    # title = "no datives"
  )



cohesion <- viz_sim %>% group_by(concept2) %>% summarize(sim = mean(similarity_Mean))

tibble(vlm_ranef) %>%
  inner_join(cohesion %>% rename(category = concept2))

lower_median <- viz_sim %>%
  mutate(
    low = similarity_Mean < median(similarity_Mean)
  ) %>% 
  group_by(concept2) %>%
  summarize(
    pct_lower = mean(low)
  )

vlm_ranef %>%
  inner_join(cohesion %>% rename(category = concept2)) %>%
  # inner_join(lower_median %>% rename(category = concept2)) %>%
  mutate(category = factor(category), category = fct_reorder(category, mean_sim)) %>%
  # ggplot(aes(category, mean_sim, fill = pct_lower, color = pct_lower)) +
  ggplot(aes(category, mean_sim, fill = sim, color = sim)) +
  # geom_col(fill = "#7570b3") +
  geom_col() +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 70, vjust = 0.6),
    axis.text = element_text(color = "black"),
    axis.title.x = element_blank(),
    plot.title = element_text(family="Inconsolata", face="bold")
  ) +
  labs(
    # y="Category-specific effect of Similarity\nrelative to global effect",
    y = "Relative Effect of Similarity"
    # title = "no datives"
  )


# ---


ns_results <- longer %>%
  filter(is_ns == TRUE) %>%
  group_by(question_id, model_setting, substitution_hop, original_arg) %>%
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
  group_by(question_id, model_setting, substitution_hop, original_arg) %>%
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

conditional <- with_ns %>%
  filter(substitution_hop == 0) %>%
  filter(correct == TRUE) %>%
  select(question_id, model_setting) %>%
  mutate(
    og_correct = TRUE
  ) %>%
  inner_join(with_ns %>% filter(substitution_hop != 0))

cat_accs <- conditional %>%
  inner_join(hypernyms %>% rename(hypernym = argument)) %>%
  group_by(model_setting, substitution_hop, hypernym) %>%
  summarize(
    outcome = mean(correct==TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    setting = str_extract(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)"),
    model = str_remove(model_setting, "(vlm_q_only_|vlm_text_|vlm_|lm_q_only_|lm_)")
  ) %>%
  filter(!setting %in% c("vlm_q_only_", "vlm_", "lm_q_only_")) %>%
  # inner_join(model_meta) %>%
  # inner_join(hypernyms %>% distinct(original_arg, substitution_hop, argument)) %>%
  filter(model %in% c("qwen2.5VL"))


cat_accs %>%
  inner_join(cohesion %>% rename(hypernym = concept2)) %>%
  ggplot(aes(sim, outcome)) +
  geom_point()