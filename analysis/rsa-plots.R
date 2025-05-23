library(tidyverse)
library(fs)
library(colorspace)

model_meta <- tribble(
  ~model, ~class, ~type,
  "Llama-3.1-8B", "llama-3.1-8b", "Text Only",
  "Llama-3.2-11B-Vision", "llama-3.1-8b", "Vision + Text",
  "Molmo-7B-D-0924", "qwen2-7b-molmo","Vision + Text",
  "Qwen2-7B", "qwen2-7b-molmo", "Text Only",
  # "Qwen2-7B", "qwen2-7b-llava-ov", "Text Only",
  "Qwen2-7B-Instruct", "qwen2-7b-instruct", "Text Only",
  "llava-1.5-7b-hf", "vicuna-7b","Vision + Text",
  "vicuna-7b-v1.5", "vicuna-7b", "Text Only",
  "llava-onevision-qwen2-7b-ov-hf", "qwen2-7b-instruct", "Vision + Text",
  "Llama-3.2-11B-Vision-Instruct", "llama-3.1.8b-instruct", "Vision + Text",
  "Llama-3.1-8B-Instruct", "llama-3.1.8b-instruct", "Text Only",
  "llava-v1.6-mistral-7b-hf", "mistral-7b", "Vision + Text",
  "Mistral-7B-Instruct-v0.2", "mistral-7b", "Text Only",
  "Qwen2.5-VL-7B-Instruct", "qwen-2.5-7b-instruct", "Vision + Text",
  "Qwen2.5-7B-Instruct", "qwen-2.5-7b-instruct", "Text Only"
)

real_model_meta <- tribble(
  ~class, ~pair,
  "llama-3.1-8b", "Llama-3.1\nvs.\nMLlama-3.2",
  "llama-3.1.8b-instruct", "Llama-3.1-I\nvs.\nMLlama-3.2-I",
  "vicuna-7b", "Vicuna\nvs.\nLlava-1.5",
  "mistral-7b", "Mistral-v0.2-I\nvs.\nLlava-Next",
  "qwen2-7b-molmo", "Qwen2\nvs.\nMolmo-D",
  "qwen2-7b-llava-ov", "Qwen2-I\nvs.\nLlava-OV",
  "qwen-2.5-7b-instruct", "Qwen2.5-I\nvs.\nQwen2.5-VL-I"
)

rsa_matrices <- dir_ls("reps/", recurse = TRUE, regexp = "*.csv") %>%
  keep(!str_detect(., "mean")) %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "reps/"),
    model = str_extract(model, "(?<=_)(.*)(?=/long-mats)"),
    matrix = case_when(
      str_detect(file, "lda.csv") ~ "model",
      TRUE ~ "wordnet"
    )
  ) %>%
  select(-file) %>% 
  inner_join(model_meta, relationship = "many-to-many") %>%
  inner_join(real_model_meta) %>%
  filter((type != "Text Only" | matrix != "wordnet")) %>%
  mutate(
    type = case_when(
      type == "Vision + Text" & matrix == "wordnet" ~ "WordNET",
      TRUE ~ type
    ),
    type = factor(type, levels = c("WordNET", "Text Only", "Vision + Text"))
  )

rsa_matrices %>%
  count(class, type, matrix)

rsa_matrices %>%
  ggplot(aes(x, y, fill = sim)) +
  geom_tile() +
  # facet_grid(class ~ type) +
  guides(fill = guide_colorbar(theme = theme(legend.key.height = unit(10, "lines")))) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  ggh4x::facet_grid2(pair ~ type, scales = "free", independent = "all") +
  # scale_fill_distiller(palette = "Greys", direction = 1) +
  scale_fill_continuous_sequential(
    palette = "Mint", 
  ) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank()
  ) +
  labs(
    fill = "Cos Sim"
  )

ggsave("plots/lda-park-etal.pdf", height = 12.32, width = 8.07, useDingbats = TRUE)
ggsave("plots/lda-park-etal.png", height = 12.32, width = 8.07, dpi = 300)
  

rsa_matrices %>%
  filter(class == "qwen-2.5-7b-instruct") %>%
  ggplot(aes(x, y, fill = sim)) +
  geom_tile() +
  # facet_grid(class ~ type) +
  # guides(fill = guide_colorbar(theme = theme(legend.key.height = unit(10, "lines")))) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  facet_wrap(~type) +
  # ggh4x::facet_grid2(class ~ type, scales = "free", independent = "all") +
  # scale_fill_distiller(palette = "Greys", direction = 1) +
  scale_fill_continuous_sequential(
    palette = "Mint", 
  ) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    # axis.text = element_blank(),
    axis.title = element_blank(),
    # legend.position = "top"
    # axis.ticks = element_blank()
  ) +
  labs(
    fill = "Cos Sim"
  )

ggsave("plots/qwen2.5-instruct-RSA.pdf", height = 3.37, width = 10.45, dpi=300,device = cairo_pdf)
