library(tidyverse)
library(fs)
library(glue)

hypernyms <- read_csv("data/gqa_entities/taxomps-hypernym.csv")
ns <- read_csv("data/gqa_entities/taxomps-ns-all.csv")
swapped <- read_csv("data/gqa_entities/taxomps-swapped.csv")

model_meta <- tribble(
  ~model, ~class, ~type,
  "SmolLM2-135M", "smollm2-135m", "Text Only",
  "SmolLM2-360M", "smollm2-360m", "Text Only",
  "SmolLM2-1.7B", "smollm2-1.7b", "Text Only",
  "SmolVLM-256M-Base", "smollm2-135m", "Vision + Text",
  "SmolVLM-500M-Base", "smollm2-360m", "Vision + Text",
  "SmolVLM-Base", "smollm2-1.7b", "Vision + Text",
  "SmolLM2-135M-Instruct", "smollm2-135m-instruct", "Text Only",
  "SmolLM2-360M-Instruct", "smollm2-360m-instruct", "Text Only",
  "SmolLM2-1.7B-Instruct", "smollm2-1.7b-instruct", "Text Only",
  "SmolVLM-256M-Instruct", "smollm2-135m-instruct", "Vision + Text",
  "SmolVLM-500M-Instruct", "smollm2-360m-instruct", "Vision + Text",
  "SmolVLM-Instruct", "smollm2-1.7b-instruct", "Vision + Text",
)

real_model_meta <- tribble(
  ~class, ~pair,
  "smollm2-135m", "SmolLM2-135M vs. SmolVLM-256M-Base",
  "smollm2-360m", "SmolLM2-360M vs. SmolVLM-500M-Base",
  "smollm2-1.7b", "SmolLM2-1.7B vs. SmolVLM-Base",
  "smollm2-135m-instruct", "SmolLM2-135M-Instruct vs. SmolVLM-256M-Instruct",
  "smollm2-360m-instruct", "SmolLM2-360M-Instruct vs. SmolVLM-500M-Instruct",
  "smollm2-1.7b-instruct", "SmolLM2-1.7B-Instruct vs. SmolVLM-Instruct",
)

model_meta %>% count(class)

read_taxomps_results <- function(subset="hypernym") {
  results <- dir_ls(glue("data/results/smollm/taxomps-{subset}-qa/"), regexp = "*.csv") %>%
    map_df(read_csv, .id = "file") %>%
    mutate(
      model = str_remove(file, glue("data/results/smollm/taxomps-{subset}-qa/")),
      model = str_extract(model, "(?<=_)(.*)(?=\\.csv)")
    ) %>%
    select(-file)
  
  return(results)
}

hypernym_results <- read_taxomps_results("hypernym") %>%
  inner_join(hypernyms) %>%
  group_by(model, category_id, parent_id) %>%
  slice_max(p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

swapped_results <- read_taxomps_results("swapped") %>%
  inner_join(swapped) %>%
  group_by(model, category_id, parent_id) %>%
  slice_max(-p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

ns_results <- read_taxomps_results("ns-all") %>%
  inner_join(ns) %>% 
  group_by(model, category_id, parent_id, ns_id) %>%
  slice_max(-p_yes, n = 1, with_ties = FALSE) %>%
  ungroup()

ns_experiment <- hypernym_results %>%
  select(model, category_id, parent_id, hypernym_label = label) %>%
  inner_join(ns_results %>% select(model, category_id, parent_id, ns_id, ns_label = label))

# strict scoring

ns_strict <- ns_experiment %>%
  group_by(model, category_id, parent_id) %>%
  summarize(
    num_correct = sum(hypernym_label == "Yes" & ns_label == "No")
  ) %>%
  ungroup() %>%
  group_by(model) %>%
  summarize(
    acc = mean(num_correct >= 4)
  ) %>%
  inner_join(model_meta, relationship = "many-to-many") %>%
  select(-model) %>%
  pivot_wider(names_from = type, values_from = acc) %>%
  mutate(
    vision_better = `Vision + Text` > `Text Only`
  )

## -- 

ns_strict %>%
  inner_join(real_model_meta) %>%
  ggplot(aes(`Text Only`, `Vision + Text`, color = pair, shape = pair, fill = pair)) +
  geom_point(size = 3) +
  geom_abline(slope = 1, linetype = "dashed", linewidth = 0.2) +
  # facet_wrap(~metric, nrow = 1) +
  scale_shape_manual(values = c(21, 22, 23, 24, 25, 8, 9)) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_x_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  theme_bw(base_size = 17, base_family = "Times") +
  theme(
    # legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "LM", y = "VLM"
  )
