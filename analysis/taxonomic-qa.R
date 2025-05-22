library(tidyverse)
library(yardstick)
library(lmerTest)
library(ggtext)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "llama-3.1-8b", "Text Only",
  "Llama-3.2-11B-Vision", "llama-3.1-8b", "Vision + Text",
  "Molmo-7B-D-0924", "qwen2-7b-molmo","Vision + Text",
  "Qwen2-7B", "qwen2-7b-molmo", "Text Only",
  "Qwen2-7B", "qwen2-7b-llava-ov", "Text Only",
  "llava-1.5-7b-hf", "vicuna-7b","Vision + Text",
  "vicuna-7b-v1.5", "vicuna-7b", "Text Only",
  "llava-onevision-qwen2-7b-ov-hf", "qwen2-7b-llava-ov", "Vision + Text",
  "Llama-3.2-11B-Vision-Instruct", "llama-3.1.8b-instruct", "Vision + Text",
  "Llama-3.1-8B-Instruct", "llama-3.1.8b-instruct", "Text Only",
  "llava-v1.6-mistral-7b-hf", "mistral-7b", "Vision + Text",
  "Mistral-7B-Instruct-v0.2", "mistral-7b", "Text Only",
  "Qwen2.5-VL-7B-Instruct", "qwen-2.5-7b-instruct", "Vision + Text",
  "Qwen2.5-7B-Instruct", "qwen-2.5-7b-instruct", "Text Only"
)

minimal_pairs <- read_csv("data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv") %>%
  janitor::clean_names()

tax_pairs <- minimal_pairs %>%
  distinct(item, category, hypernym)

results <- fs::dir_ls("data/results/hypernym-minimal-pairs-qa/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/hypernym-minimal-pairs-qa/"),
    model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
  ) %>%
  select(-file) %>%
  inner_join(minimal_pairs)

correctness <- results %>%
  mutate(
    ns_correct = hypernym_pred == "Yes" & negative_pred != "Yes",
    swapped_correct = hypernym_pred == "Yes" & swapped_pred != "Yes",
    both_correct = hypernym_pred == "Yes" & negative_pred != "Yes" & swapped_pred != "Yes"
  )

accuracy <- correctness %>%
  group_by(model) %>%
  summarize(
    ns_correct = mean(ns_correct),
    swapped_correct = mean(swapped_correct),
    both_correct = mean(both_correct)
  )

set.seed(1024)
bootstrapped <- correctness %>%
  select(item, model, ns_correct, swapped_correct, both_correct) %>%
  pivot_longer(ns_correct:both_correct, names_to = "measure", values_to = "correct") %>%
  group_by(model, measure) %>%
  nest() %>%
  mutate(
    booted = map(data, function(x) {
      Hmisc::smean.cl.boot(x$correct, conf.int = .95, B = 5000) %>% bind_rows()
    })
  ) %>%
  select(-data) %>%
  unnest(booted) %>%
  ungroup() %>%
  mutate(
    measure = case_when(
      str_detect(measure, "ns_") ~ "negative-sample",
      str_detect(measure, "swapped_") ~ "swapped",
      TRUE ~ "overall"
    ),
    measure = factor(measure, levels = c("negative-sample", "swapped", "overall"))
  )

bootstrapped %>%
  inner_join(model_meta, relationship = "many-to-many") %>% View()
  ggplot(aes(class, Mean, color = type, fill = type)) +
  geom_col(position=position_dodge2(0.5), width = 0.5) +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width = 0.2, position=position_dodge(0.5), color = "black") +
  # geom_hline(yintercept = 0.20, linetype = "dashed") +
  facet_wrap(~ measure) +
  scale_y_continuous(limits = c(0, 1.0), expand = c(0.01, 0)) +
  scale_color_manual(values = c("#d8b365", "#5ab4ac"), aesthetics = c("color", "fill")) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 20, hjust = 1.0)
  ) +
  labs(
    x = "Model Family",
    y = "Accuracy",
    color = "Modality",
    fill = "Modality"
  )

