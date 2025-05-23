library(tidyverse)
library(e1071)
library(ggtext)
library(patchwork)
# library(plotly)
# library(kernlab)
# library(svmpath)

# plot_ly(pcax=temp, y=pressure, z=dtime, type="scatter3d", mode="markers", color=temp)
# plot_ly(pca_results %>% filter(model_class == "LM"), x = ~x1, y = ~x2, z = ~x3, color = ~subs_type)



pca_results <- fs::dir_ls("pca-data/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_replace(model, "pca-data/", ""),
    model = str_replace(model, ".csv", ""),
    model_class = case_when(
      str_detect(model, "vision") ~ "VLM",
      TRUE ~ "LM"
    )
  ) %>%
  rename(x1 = x, x2 = y, x3 = z) %>%
  mutate(
    y = case_when(subs_type == "hypernym" ~ 1, TRUE ~ -1),
    y = factor(y)
  )


fit_svm <- function(df, cost = 1) {
  svm_model <- svm(y~ x1 + x2, 
                   data = df, 
                   cost = cost,
                   type = "C-classification",
                   kernel = "linear", 
                   scale = F)
  w <- t(svm_model$coefs) %*% svm_model$SV
  slope_1 <- -w[1]/w[2]
  slope_1
  
  intercept_1 <- svm_model$rho/w[2]
  
  # df1 = df %>% filter(subset == "vlm-better")
  # df2 = df %>% filter(subset == "equal")
  
  # Get decision values (distance from hyperplane)
  # decision_values1 <- attributes(predict(svm_model, df1[c("x1", "x2")], decision.values = TRUE))$decision.values
  # decision_values2 <- attributes(predict(svm_model, df2[c("x1", "x2")], decision.values = TRUE))$decision.values
  
  decision_values <- attributes(predict(svm_model, df[c("x1", "x2")], decision.values = TRUE))$decision.values
  
  # True labels as +1 or -1
  # y_true1 <- as.numeric(as.character(df1$y))
  # y_true2 <- as.numeric(as.character(df2$y))
  y_true <- as.numeric(as.character(df$y))
  
  # Functional margins: y_i * f(x_i)
  # functional_margins1 <- y_true1 * decision_values1
  # functional_margins2 <- y_true2 * decision_values2
  functional_margins <- y_true * decision_values
  
  # total_samples1 = length(functional_margins1)
  # total_samples2 = length(functional_margins2)
  total_samples = length(functional_margins)
  
  # misclassification1 <- length(which(functional_margins1 <= 0))/total_samples1
  # margin_error1 <- length(which(functional_margins1 > 0 & functional_margins1 < 1))/total_samples1
  # misclassification2 <- length(which(functional_margins2 <= 0))/total_samples2
  # margin_error2 <- length(which(functional_margins2 > 0 & functional_margins2 < 1))/total_samples2
  misclassification <- length(which(functional_margins <= 0))/total_samples
  margin_error <- length(which(functional_margins > 0 & functional_margins < 1))/total_samples
  
  # svm_error <- functional_margins < 1
  # 
  # num_margin_errors <- len
  # svm_error1 <- length(which(functional_margins1 < 1)) / total_samples1
  # svm_error2 <- length(which(functional_margins2 < 1)) / total_samples2
  svm_error <- length(which(functional_margins < 1)) / total_samples
  
  # stopifnot(misclassification + margin_error == svm_error)
  
  # return(
  #   tibble(
  #     slope = c(slope_1, slope_1), 
  #     intercept = c(intercept_1, intercept_1),
  #     interceptlow = c(intercept_1 - 1/w[2], intercept_1 - 1/w[2]), 
  #     intercepthigh = c(intercept_1 + 1/w[2], intercept_1 + 1/w[2]),
  #     svm_error = c(svm_error1, svm_error2),
  #     classification_error = c(misclassification1, misclassification2),
  #     margin_error = c(margin_error1, margin_error2),
  #     subset = c("vlm-better", "equal"),
  #     setting = c("VLM > LM", "VLM ~ LM")
  #   )
  # )
  
  return(
    tibble(
      slope = slope_1, 
      intercept = intercept_1,
      interceptlow = intercept_1 - 1/w[2], 
      intercepthigh = intercept_1 + 1/w[2],
      svm_error = svm_error,
      classification_error = misclassification,
      margin_error = margin_error
      # subset = c("vlm-better", "equal"),
      # setting = c("VLM > LM", "VLM ~ LM")
    )
  )
  
  # # Get decision values (distance from hyperplane)
  # decision_values <- attributes(predict(svm_model, df[c("x1", "x2")], decision.values = TRUE))$decision.values
  # 
  # # True labels as +1 or -1
  # y_true <- as.numeric(as.character(df$y))
  # 
  # # Functional margins: y_i * f(x_i)
  # functional_margins <- y_true * decision_values
  # 
  # total_samples = length(functional_margins)
  # 
  # misclassification <- length(which(functional_margins <= 0))/total_samples
  # margin_error <- length(which(functional_margins > 0 & functional_margins < 1))/total_samples
  # 
  # # svm_error <- functional_margins < 1
  # # 
  # # num_margin_errors <- len
  # svm_error <- length(which(functional_margins < 1)) / total_samples
  # 
  # # stopifnot(misclassification + margin_error == svm_error)
  # 
  # return(
  #   list(
  #     slope = slope_1, 
  #     intercept = intercept_1, 
  #     interceptlow = intercept_1 - 1/w[2], 
  #     intercepthigh = intercept_1 + 1/w[2],
  #     svm_error = svm_error,
  #     classification_error = misclassification,
  #     margin_error = margin_error
  #   )
  # )
  
}
  
vision_fit <- fit_svm(pca_results %>% filter(model_class == "VLM"), cost = 0.2)
text_fit <- fit_svm(pca_results %>% filter(model_class == "LM"), cost = 0.2)

vision_fit
text_fit


# vlm_better_vlm <- fit_svm(pca_results %>% filter(subset == "vlm-better", model_class == "Vision + Text"))
# vlm_better_text <- fit_svm(pca_results %>% filter(subset == "vlm-better", model_class == "Text Only"))
# equal_vlm <- fit_svm(pca_results %>% filter(subset == "equal", model_class == "Vision + Text"))
# equal_text <- fit_svm(pca_results %>% filter(subset == "equal", model_class == "Text Only"))


# ab_line_vlm_better <- tibble(
#   slope = c()
# )

# fit_svm(pca_results %>% filter(model_class == "Vision + Text"))
# 
# fit_svm(pca_results %>% filter(model_class == "Text Only"))

# ablines <- bind_rows(
#   as_tibble(vlm_better_text) %>% mutate(model_class = "Text Only", subset = "vlm-better", setting = "VLM > LM"),
#   as_tibble(vlm_better_vlm) %>% mutate(model_class = "Vision + Text", subset = "vlm-better", setting = "VLM > LM"),
#   as_tibble(equal_text) %>% mutate(model_class = "Text Only", subset = "equal", setting = "VLM ~ LM"),
#   as_tibble(equal_vlm) %>% mutate(model_class = "Vision + Text", subset = "equal", setting = "VLM ~ LM")
# ) %>%
#   mutate(
#     model_stats = glue::glue("{model_class} (<b>SVM-Error:</b> {round(svm_error, 2)})")
#   )

ablines <- bind_rows(
  text_fit %>% mutate(model_class = "LM") %>% mutate(acc = "0.80"),
  vision_fit %>% mutate(model_class = "VLM") %>% mutate(acc = "0.95"),
  # as_tibble(vlm_better_vlm) %>% mutate(model_class = "Vision + Text", subset = "vlm-better", setting = "VLM > LM"),
  # as_tibble(equal_text) %>% mutate(model_class = "Text Only", subset = "equal", setting = "VLM ~ LM"),
  # as_tibble(equal_vlm) %>% mutate(model_class = "Vision + Text", subset = "equal", setting = "VLM ~ LM")
) %>%
  mutate(
    model_stats = glue::glue("{model_class} (<b>SVM-Error:</b> {round(svm_error, 2)})")
  )

pca_results %>%
  mutate(
    correct = factor(correct, levels = c(TRUE, FALSE)),
    model_stats = case_when(
      model_class == "LM" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.47)"),
      model_class == "VLM" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.39)")
    ),
    subs_type = str_to_title(subs_type),
    subs_type = factor(subs_type, levels = c("Hypernym","Negative"))
  ) %>%
  ggplot(aes(x1, x2)) +
  geom_point(aes(color = subs_type, shape = subs_type), size = 2.5, alpha = 0.2) +
  facet_wrap(~model_stats, scales = "free") +
  geom_abline(data = ablines, aes(slope = slope, intercept = intercept)) +
  geom_abline(data = ablines, aes(slope = slope, intercept = interceptlow), linetype = "dashed") +
  geom_abline(data = ablines, aes(slope = slope, intercept = intercepthigh), linetype = "dashed") +
  scale_color_brewer(palette = "Dark2", direction = 1) +
  scale_shape_manual(values = c(16, 4)) +
  # scale_shape_manual(values = c(4, 16)) +
  theme_bw(base_size = 17, base_family = "Times") +
  guides(color = guide_legend(override.aes = list(alpha = 0.8))) +
  theme(
    panel.grid = element_blank(),
    legend.position = "top",
    strip.text = element_markdown(color = "black", size = 15, padding = unit(3, "points")),
    # strip.text = element_markdown(color = "black", size = 15,),
    axis.text = element_markdown(color = "black"),
    plot.title = element_text(hjust = 0.5)
    # axis.ticks = element_line()
  ) +
  labs(
    x = "PC1", y = "PC2", color = "Substitution",
    shape = "Substitution"
  )

ggsave("plots/pca-qwen-full.pdf", width = 8.43, height = 4.97, dpi = 300, device=cairo_pdf)



