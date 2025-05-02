library(tidyverse)

edge_accuracies <- read_csv("data/results/model_results - edge_acc.csv") %>%
  janitor::clean_names() %>%
  mutate(
    concept1_count = as.numeric(str_extract(raw_counts, "(?<=\\()(.*)(?=\\,)")),
    concept2_count = as.numeric(str_extract(raw_counts, "(?<=,\\s)(.*)(?=\\))"))
  ) %>%
  select(-raw_counts)

sims <- read_csv("data/results/llava-hf_llava-1.5-7b-hf_similarity_scores.csv") %>%
  janitor::clean_names()

edge_accuracies %>%
  filter(model == "llava", model_type == "vlm-text") %>%
  inner_join(
    sims %>% select(concept1, concept2, sim_lhs_pooled)
  ) %>%
  filter(concept2_count > 10) %>%
  ggplot(aes(sim_lhs_pooled, accuracy)) +
  geom_point() +
  facet_wrap(~concept2)
