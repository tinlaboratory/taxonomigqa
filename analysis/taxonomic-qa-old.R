library(tidyverse)
library(yardstick)
library(lmerTest)
library(ggtext)

minimal_pairs <- read_csv("data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv") %>%
  janitor::clean_names()

tax_pairs <- minimal_pairs %>%
  distinct(item, category, hypernym)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "Llama-3.1-8B", "Text Only",
  "Llama-3.2-11B-Vision", "Llama-3.1-8B", "Vision + Text",
  "Molmo-7B-D-0924", "Qwen2-7B","Vision + Text",
  "Qwen2-7B", "Qwen2-7B", "Text Only",
  "llava-1.5-7b-hf", "Vicuna-7B-v1.5","Vision + Text",
  "vicuna-7b-v1.5", "Vicuna-7B-v1.5", "Text Only"
)

results <- fs::dir_ls("data/results/hypernym-minimal-pairs-qa/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/hypernym-minimal-pairs-qa/"),
    model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
  ) %>%
  select(-file) %>%
  inner_join(minimal_pairs)

# sim_results <- fs::dir_ls("data/results/hypernym-qa-sims/", regexp = "*.csv") %>%
#   map_df(read_csv, .id = "file") %>%
#   mutate(
#     model = str_remove(file, "data/results/hypernym-qa-sims/"),
#     model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
#   ) %>%
#   select(-file) %>%
#   inner_join(tax_pairs)

# tax_results <- results %>%
#   distinct(model, item, hypernym_pred, hypernym_yes)

# tax only

# tax_results %>%
#   group_by(model) %>%
#   summarize(
#     accuracy = mean(hypernym_pred == "Yes")
#   )


results %>% count(model)

correctness <- results %>%
  group_by(model, category, hypernym) %>%
  summarize(
    num_correct = sum(hypernym_pred == "Yes" & negative_pred == "No")
  ) %>%
  ungroup() 

correctness %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 4)
  )

set.seed(1024)
bootstrapped <- correctness %>%
  mutate(correct = num_correct>=4) %>%
  group_by(model) %>%
  nest() %>%
  mutate(
    booted = map(data, function(x) {
      Hmisc::smean.cl.boot(x$correct, conf.int = .95, B = 5000) %>% bind_rows()
    })
  ) %>%
  select(-data) %>%
  unnest(booted) %>%
  ungroup()

bootstrapped %>%
  inner_join(model_meta) %>%
  ggplot(aes(class, Mean, color = type, fill = type)) +
  geom_col(position=position_dodge2(0.5), width = 0.5) +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width = 0.2, position=position_dodge(0.5), color = "black") +
  # geom_hline(yintercept = 0.20, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1.0), expand = c(0.01, 0)) +
  scale_color_manual(values = c("#d8b365", "#5ab4ac"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Model Family",
    y = "Accuracy",
    color = "Modality",
    fill = "Modality"
  )

real_predictions = factor(c(rep("Yes", 1980), rep("No", 5940)), levels = c("No", "Yes"))
all_no = factor(c(rep("No", 7920)), levels = c("No", "Yes"))

f_meas_vec(real_predictions, all_no)

pred_sim <- sim_results %>%
  inner_join(tax_results) %>%
  inner_join(model_meta) %>%
  filter(!str_detect(model, "Llama"))

diff_results <- pred_sim %>%
  select(item, layer, class, type, sim, hypernym_yes) %>%
  pivot_wider(names_from = type, values_from = c(sim, hypernym_yes)) %>%
  janitor::clean_names() %>%
  mutate(
    sim_diff = sim_vision_text - sim_text_only,
    yes_diff = hypernym_yes_vision_text - hypernym_yes_text_only
  ) 

fit_results <- diff_results %>%
  group_by(class, layer) %>%
  nest() %>%
  mutate(
    fit = map(data, function(x) {
      fit <- lm(yes_diff ~ sim_diff, data = x) 
      
      estimate <- fit %>%
        broom::tidy(conf.int = TRUE) %>%
        filter(term == "sim_diff")
      
      r_sq = fit %>%
        broom::glance() %>%
        pull(r.squared)
      
      return(estimate %>% mutate(r_sq = r_sq))
    })
  ) %>%
  select(-data) %>%
  unnest(cols = c(fit))

# p.adjust(fit_results$p.value)
fit_results %>%
  ungroup() %>%
  nest() %>%
  mutate(
    adj_p = map(data, function(x) {
      p.adjust(x$p.value, "BH")
    })
  ) %>%
  unnest(cols = c(data, adj_p)) %>%
  mutate(significance = adj_p < 0.01) %>%
  filter(layer != 0) %>%
  ggplot(aes(layer, estimate, color = significance, fill = significance, shape = significance)) +
  geom_point(size = 2.5) +
  geom_linerange(aes(ymin = conf.low, ymax = conf.high)) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_shape_manual(values = c(4, 19)) +
  scale_color_manual(values = c("red", "black")) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.title.y = element_markdown()
  ) +
  labs(
    x = "Layer",
    y = "Coefficient for &Delta;<sub><i>sim</i></sub>"
  )

# 402 w 374 h

fit_results %>%
  filter(layer != 0) %>%
  ggplot(aes(layer, sqrt(r_sq))) +
  geom_point()

summary(fit)

confint(fit)

broom::tidy(fit, conf.int = TRUE)
