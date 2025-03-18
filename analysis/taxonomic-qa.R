library(tidyverse)
library(yardstick)

minimal_pairs <- read_csv("data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv")

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

results %>% count(model)

correctness <- results %>%
  group_by(model, category, hypernym) %>%
  summarize(
    num_correct = sum(hypernym_pred == "Yes" & negative_pred == "No")
  ) %>%
  ungroup() 

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
