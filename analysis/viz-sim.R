library(tidyverse)
library(lmerTest)


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
    pct_lower = mean(low),
    avg_sim = mean(similarity_Mean)
  )

# avg_sims <- viz_sim %>%
#   group_by(concept2) %>%
#   summarize(
#     sim = mean(similarity_Mean)
#   )

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
  group_by(model_setting, hypernym) %>%
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
  inner_join(real_model_meta) %>%
  select(-model_setting, -model, -class, -type) %>%
  pivot_wider(names_from = setting, values_from = outcome, values_fill = 0) %>%
  janitor::clean_names()
  # inner_join(hypernyms %>% distinct(original_arg, substitution_hop, argument)) %>%
  # filter(model %in% c("qwen2.5VL"))

cat_accs %>% 
  filter(pair == "Qwen2.5-I vs. Qwen2.5-VL-I") %>%
  mutate(
    diff = vlm_text - lm,
    alt_metric = case_when(
      vlm_text > lm ~ vlm_text,
      lm > vlm_text ~ lm,
      TRUE ~ 0
    ),
    color = case_when(
      vlm_text > lm ~ "#7570b3",
      lm > vlm_text ~ "#e6ab02",
      TRUE ~ "black"
    ),
    hypernym = factor(hypernym),
    hypernym = fct_reorder(hypernym, diff)
  ) %>% 
  filter(diff!=0) %>%
  ggplot(aes(hypernym, diff, color = color, fill = color)) +
  geom_col() +
  facet_wrap(~pair) +
  scale_color_identity(aesthetics = c("fill", "color")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    axis.text = element_text(color = "black"),
    axis.title.x = element_blank(),
    plot.title = element_text(family="Inconsolata", face="bold")
  )
  

vlm_ranef %>%
  inner_join(
    cat_accs %>% 
      filter(pair == "Qwen2.5-I vs. Qwen2.5-VL-I") %>%
      mutate(
        diff = vlm_text - lm,
        alt_metric = case_when(
          vlm_text > lm ~ vlm_text,
          lm > vlm_text ~ lm,
          TRUE ~ 0
        )
      ) %>%
      rename(category = hypernym)
  ) %>% as_tibble() %>%
  # filter(diff > 0) %>%
  mutate(category = factor(category), category = fct_reorder(category, mean_sim)) %>%
  ggplot(aes(category, mean_sim)) +
    geom_col(fill = "#7570b3") +
    theme_bw(base_size = 16, base_family = "Times") +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      axis.text = element_text(color = "black"),
      axis.title.x = element_blank(),
      plot.title = element_text(family="Inconsolata", face="bold")
    ) +
    labs(
      # y="Category-specific effect of Similarity\nrelative to global effect",
      y = "Relative Effect of Similarity"
      # title = "no datives"
    )


joined_reg_new <- joined %>%
  mutate(
    mean_sim = similarity_Mean,
    pairwise_sim = similarity_pairwise,
    category = factor(category),
    type = case_when(
      type == "lm" ~ -1,
      TRUE ~ 1
    )
  ) %>%
  inner_join(qwen_diffs) %>%
  filter(!is.na(diff)) %>%
  # inner_join(cat_accs %>%
  #              filter(pair == "Qwen2.5-I vs. Qwen2.5-VL-I") %>%
  #              mutate(
  #                diff = vlm_text - lm,
  #                alt_metric = case_when(
  #                  vlm_text > lm ~ vlm_text,
  #                  lm > vlm_text ~ lm,
  #                  TRUE ~ 0
  #                )
  #              ) %>%
  #              rename(concept2 = hypernym)) %>%
  filter(diff > 0)

fit <- lmer(vlm_text ~ mean_sim + (1 + mean_sim | concept2),REML = F, data = joined_reg_new %>% filter(type == 1))
fit_fe <- lmer(vlm_text ~  (1 + mean_sim | concept2),REML = F, data = joined_reg_new %>% filter(type == 1))

anova(fit, fit_fe)

# summary(fit)

fit2 <- lmer(lm ~ mean_sim + (1 + mean_sim | concept2),REML = F, data = joined_reg_new %>% filter(type == 1))


summary(fit)

fit_no_sim <- lmer(accuracy ~  (1 + mean_sim || concept2),REML = F, data = joined_reg_new %>% filter(type == 1))
fit_no_sim_all <- lmer(accuracy ~  (1 || concept2),REML = F, data = joined_reg_new %>% filter(type == 1))

summary(fit)
# anova(fit, fit_no_sim, fit_no_sim_all)
# 
# fit <- lmer(accuracy ~ mean_sim + (1 + mean_sim | concept2),REML = F, data = joined_reg %>% filter(type == 1))
# fit_no_slope <- lmer(accuracy ~ mean_sim + (1  | concept2),REML = F, data = joined_reg %>% filter(type == 1))
# 
# anova(fit, fit_no_slope)
# summary(fit)

ranef(fit)

coef(fit)

stats <- viz_sim %>%
  mutate(
    high = similarity_Mean >= median(similarity_Mean),
    low = similarity_Mean > median(similarity_Mean),
    diff_med = similarity_Mean - median(similarity_Mean)
  ) %>% 
  group_by(concept2) %>%
  summarize(
    pct_lower = mean(low),
    pct_higher = mean(high),
    avg_sim = mean(similarity_Mean),
    diff_med = mean(diff_med)
  )

vlm_ranef <- rownames_to_column(ranef(fit)$concept2, var = "category") %>%
  # mutate(verb = factor(category, levels = fcts)) %>%
  mutate(category = factor(category), category = fct_reorder(category, mean_sim))

vlm_ranef %>%
  inner_join(stats %>% rename(category = concept2)) %>%
  mutate(category = factor(category), category = fct_reorder(category, mean_sim)) %>%
  ggplot(aes(category, mean_sim, fill = pct_higher, color = pct_higher)) +
  geom_col() +
  scale_color_gradient(high = "#132B43", low = "#56B1F7", aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    axis.text = element_text(color = "black"),
    axis.title.x = element_blank(),
    plot.title = element_text(family="Inconsolata", face="bold")
  ) +
  labs(
    # y="Category-specific effect of Similarity\nrelative to global effect",
    y = "Relative Effect of Similarity",
    color = "% Higher\nthan median",
    fill = "% Higher\nthan median",
    # title = "no datives"
  )

ggsave("plots/relative_effects_qwen_vlm.pdf", width = 13.06, height = 4.49, dpi = 300, device = cairo_pdf)
ggsave("plots/relative_effects_qwen_vlm.svg", width = 13.06, height = 4.49, dpi = 300)

# cat_accs %>%
#   ggplot(aes(lm, vlm_text, color = pair, shape = pair, fill = pair)) +
#   geom_point(size = 3) +
#   geom_abline(slope = 1, linetype = "dashed", linewidth = 0.2) +
#   # facet_wrap(~metric, nrow = 1) +
#   scale_shape_manual(values = c(21, 22, 23, 24, 25, 8, 9)) +
#   scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
#   scale_x_continuous(limits = c(0,1), labels = scales::percent_format()) +
#   scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
#   theme_bw(base_size = 16, base_family = "Times") +
#   theme(
#     # legend.position = "top",
#     legend.title = element_blank(),
#     legend.text = element_text(size = 12),
#     axis.text = element_text(color = "black")
#   ) +
#   labs(
#     x = "LM", y = "VLM"
#   )


# cat_accs %>%
#   inner_join(cohesion %>% rename(hypernym = concept2)) %>%
#   ggplot(aes(sim, outcome)) +
#   geom_point()