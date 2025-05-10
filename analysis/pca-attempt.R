library(tidyverse)

pca_results <- fs::dir_ls("pca-data/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_replace(model, "pca-data/", ""),
    model = str_replace(model, ".csv", "")
  )

pca_results %>%
  ggplot(aes(x, y, color = subs_type, shape = correct)) +
  geom_point(size = 2.5, alpha = 0.8) +
  facet_wrap(~type, scales = "free") +
  scale_shape_manual(values = c(4, 16)) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top"
  )
