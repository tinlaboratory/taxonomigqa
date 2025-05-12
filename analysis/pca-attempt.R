library(tidyverse)
library(e1071)
library(ggtext)
library(patchwork)
# library(kernlab)
# library(svmpath)

pca_results <- fs::dir_ls("pca-data/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_replace(model, "pca-data/", ""),
    model = str_replace(model, ".csv", ""),
    model_class = case_when(
      str_detect(model, "vision") ~ "Vision + Text",
      TRUE ~ "Text Only"
    ),
    setting = case_when(
      str_detect(subset, "vlm-better") ~ "VLM > LM",
      TRUE ~ "VLM ~ LM"
    )
  ) %>%
  rename(x1 = x, x2 = y) %>%
  mutate(
    y = case_when(subs_type == "hypernym" ~ 1, TRUE ~ -1),
    y = factor(y)
  )

pca_results %>%
  ggplot(aes(x1, x2, color = subs_type)) +
  geom_point(aes(shape = correct), size = 2.5, alpha = 0.8) +
  stat_ellipse() +
  # facet_wrap(~type, scales = "free") +
  # facet_grid(model_class ~ setting) +
  ggh4x::facet_grid2(model_class ~ setting, scales = "free", independent = "all") +
  scale_shape_manual(values = c(4, 16)) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    legend.position = "top"
  )


fit_svm <- function(df) {
  svm_model <- svm(y~ x1 + x2, 
                   data = df, 
                   type = "C-classification",
                   kernel = "linear", 
                   scale = F)
  w <- t(svm_model$coefs) %*% svm_model$SV
  slope_1 <- -w[1]/w[2]
  slope_1
  
  intercept_1 <- svm_model$rho/w[2]
  
  df1 = df %>% filter(subset == "vlm-better")
  df2 = df %>% filter(subset == "equal")
  
  # Get decision values (distance from hyperplane)
  decision_values1 <- attributes(predict(svm_model, df1[c("x1", "x2")], decision.values = TRUE))$decision.values
  decision_values2 <- attributes(predict(svm_model, df2[c("x1", "x2")], decision.values = TRUE))$decision.values
  
  # True labels as +1 or -1
  y_true1 <- as.numeric(as.character(df1$y))
  y_true2 <- as.numeric(as.character(df2$y))
  
  # Functional margins: y_i * f(x_i)
  functional_margins1 <- y_true1 * decision_values1
  functional_margins2 <- y_true2 * decision_values2
  
  total_samples1 = length(functional_margins1)
  total_samples2 = length(functional_margins2)
  
  misclassification1 <- length(which(functional_margins1 <= 0))/total_samples1
  margin_error1 <- length(which(functional_margins1 > 0 & functional_margins1 < 1))/total_samples1
  misclassification2 <- length(which(functional_margins2 <= 0))/total_samples2
  margin_error2 <- length(which(functional_margins2 > 0 & functional_margins2 < 1))/total_samples2
  
  # svm_error <- functional_margins < 1
  # 
  # num_margin_errors <- len
  svm_error1 <- length(which(functional_margins1 < 1)) / total_samples1
  svm_error2 <- length(which(functional_margins2 < 1)) / total_samples2
  
  # stopifnot(misclassification + margin_error == svm_error)
  
  return(
    tibble(
      slope = c(slope_1, slope_1), 
      intercept = c(intercept_1, intercept_1),
      interceptlow = c(intercept_1 - 1/w[2], intercept_1 - 1/w[2]), 
      intercepthigh = c(intercept_1 + 1/w[2], intercept_1 + 1/w[2]),
      svm_error = c(svm_error1, svm_error2),
      classification_error = c(misclassification1, misclassification2),
      margin_error = c(margin_error1, margin_error2),
      subset = c("vlm-better", "equal"),
      setting = c("VLM > LM", "VLM ~ LM")
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
  
vision_fit <- fit_svm(pca_results %>% filter(model_class == "Vision + Text"))
text_fit <- fit_svm(pca_results %>% filter(model_class == "Text Only"))


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
  text_fit %>% mutate(model_class = "Text Only"),
  vision_fit %>% mutate(model_class = "Vision + Text"),
  # as_tibble(vlm_better_vlm) %>% mutate(model_class = "Vision + Text", subset = "vlm-better", setting = "VLM > LM"),
  # as_tibble(equal_text) %>% mutate(model_class = "Text Only", subset = "equal", setting = "VLM ~ LM"),
  # as_tibble(equal_vlm) %>% mutate(model_class = "Vision + Text", subset = "equal", setting = "VLM ~ LM")
) %>%
  mutate(
    model_stats = glue::glue("{model_class} (<b>SVM-Error:</b> {round(svm_error, 2)})")
  )



vlm_better_plot <- pca_results %>%
  filter(subset == "vlm-better") %>%
  mutate(
    correct = factor(correct, levels = c(TRUE, FALSE)),
    model_stats = case_when(
      model_class == "Text Only" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.37)"),
      model_class == "Vision + Text" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.29)")
    ),
    subs_type = str_to_title(subs_type)
  ) %>%
  ggplot(aes(x1, x2)) +
  geom_point(aes(color = subs_type), size = 2.5, alpha = 0.7) +
  facet_wrap(~model_stats, scales = "free") +
  geom_abline(data = ablines %>% filter(subset == "vlm-better"), aes(slope = slope, intercept = intercept)) +
  geom_abline(data = ablines %>% filter(subset == "vlm-better"), aes(slope = slope, intercept = interceptlow), linetype = "dashed") +
  geom_abline(data = ablines %>% filter(subset == "vlm-better"), aes(slope = slope, intercept = intercepthigh), linetype = "dashed") +
  scale_color_brewer(palette = "Dark2") +
  scale_shape_manual(values = c(16, 4)) +
  theme_minimal(base_size = 16, base_family = "Times") +
  theme(
    # panel.grid = element_blank(),
    legend.position = "bottom",
    strip.text = element_markdown(color = "black", size = 15, padding = unit(3, "points")),
    axis.text = element_markdown(color = "black"),
    plot.title = element_text(hjust = 0.5)
    # axis.ticks = element_line()
  ) +
  labs(
    x = "PC1", y = "PC2", color = "Substitution",
    # shape = "Correct?",
    title = "VLM > LM"
  )


equal_plot <- pca_results %>%
  filter(subset == "equal") %>%
  mutate(
    correct = factor(correct, levels = c(TRUE, FALSE)),
    model_stats = case_when(
      model_class == "Text Only" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.63)"),
      model_class == "Vision + Text" ~ glue::glue("{model_class} (<b>SVM-Error:</b> 0.48)")
    ),
    subs_type = str_to_title(subs_type)
  ) %>%
  ggplot(aes(x1, x2)) +
  geom_point(aes(color = subs_type), size = 2.5, alpha = 0.7) +
  facet_wrap(~model_stats, scales = "free") +
  geom_abline(data = ablines %>% filter(subset == "equal"), aes(slope = slope, intercept = intercept)) +
  geom_abline(data = ablines %>% filter(subset == "equal"), aes(slope = slope, intercept = interceptlow), linetype = "dashed") +
  geom_abline(data = ablines %>% filter(subset == "equal"), aes(slope = slope, intercept = intercepthigh), linetype = "dashed") +
  scale_color_brewer(palette = "Dark2") +
  scale_shape_manual(values = c(16, 4)) +
  theme_minimal(base_size = 16, base_family = "Times") +
  theme(
    # panel.grid = element_blank(),
    legend.position = "bottom",
    strip.text = element_markdown(color = "black", size = 15, padding = unit(3, "points")),
    axis.text = element_markdown(color = "black"),
    plot.title = element_text(hjust = 0.5)
    # axis.ticks = element_line()
  ) +
  labs(
    x = "PC1", y = "PC2", color = "Substitution",
    # shape = "Correct?",
    title = "VLM ~ LM"
  )


vlm_better_plot + equal_plot + plot_layout(guides = "collect", axis_titles = "collect") & theme(legend.position = 'bottom')

ggsave("plots/pca-qwen.pdf", width = 14.42, height = 5.08, dpi = 300, device=cairo_pdf)
  

# geom_abline(slope = vlm_better_vlm,
  #             intercept = intercept_1) +
  # geom_abline(slope = slope_1,
  #             intercept = intercept_1 - 1/w[2],
  #             linetype = "dashed") +
  # geom_abline(slope = slope_1,
  #             intercept = intercept_1 + 1/w[2],
  #             linetype = "dashed")







# --------

vlm_better_vlm <- pca_results %>%
  # filter(subset == "equal", model_class == "Vision + Text") %>%
  # filter(subset == "equal", model_class == "Text Only") %>%
  # filter(subset == "vlm-better", model_class == "Text Only") %>%
  # filter(subset == "vlm-better", model_class == "Vision + Text") %>%
  rename(x1 = x, x2 = y) %>%
  mutate(
    y = case_when(subs_type == "hypernym" ~ 1, TRUE ~ -1),
    y = factor(y)
  )

svm_model <- svm(y~ x1 + x2, 
                 data = vlm_better_vlm, 
                 type = "C-classification",
                 # cost = 1e6,
                 kernel = "linear", 
                 scale = F)
svm_model

pred_test <- predict(svm_model, vlm_better_vlm[c("x1", "x2")])
mean(pred_test == vlm_better_vlm$y)

# find slope and intercept of the boundary
# build the weight vector, `w`, from `coefs` and `SV` elements of `svm_model`
w <- t(svm_model$coefs) %*% svm_model$SV
w # weight vector

w


# slope
slope_1 <- -w[1]/w[2]
slope_1

intercept_1 <- svm_model$rho/w[2]

vlm_better_vlm %>%
  ggplot(aes(x1, x2)) +
  geom_point(aes(color = subs_type, shape = subs_type)) +
  scale_color_brewer(palette = "Dark2") +
  geom_abline(slope = slope_1,
              intercept = intercept_1) +
  geom_abline(slope = slope_1,
              intercept = intercept_1 - 1/w[2],
              linetype = "dashed") +
  geom_abline(slope = slope_1,
              intercept = intercept_1 + 1/w[2],
              linetype = "dashed")

# Get decision values (distance from hyperplane)
decision_values <- attributes(predict(svm_model, vlm_better_vlm[c("x1", "x2")], decision.values = TRUE))$decision.values

# True labels as +1 or -1
# y_true <- ifelse(vlm_better_vlm$y == levels(vlm_better_vlm$y)[1], -1, 1)
y_true <- as.numeric(as.character(vlm_better_vlm$y))

# Functional margins: y_i * f(x_i)
functional_margins <- y_true * decision_values

length(which(functional_margins <= 0))

inside_margin <- which(functional_margins > 0 & functional_margins < 1)

margin_errors <- functional_margins < 1

num_margin_errors <- sum(margin_errors)
total_samples <- length(margin_errors)
margin_error_rate <- num_margin_errors / total_samples

margin_error_rate

length(inside_margin)/total_samples

decision_vals <- attr(predict(svm_model, vlm_better_vlm[c("x1", "x2")], decision.values = TRUE), "decision.values")

# Convert factor labels to numeric {-1, +1}
true_y <- as.numeric(as.character(vlm_better_vlm$subs_type))

# fit_hmc <- ksvm(  # use ksvm() to find the OSH
#   x = data.matrix(vlm_better_vlm[c("x1", "x2")]),
#   y = as.factor(vlm_better_vlm$subs_type), 
#   kernel = "vanilladot",  # no fancy kernel, just ordinary dot product
#   C = Inf,                # to approximate hard margin classifier
#   prob.model = TRUE,       # needed to obtain predicted probabilities
#   minsteps = 1000000
# )
# 
# npts <- 500
# xgrid <- expand.grid(
#   x1 = seq(from = -40, 60, length = npts),
#   x2 = seq(from = -40, 60, length = npts)
# )
# 
# 
# prob_hmc <- predict(fit_hmc, newdata = data.matrix(xgrid), type = "probabilities")
# kernlab::predict(fit_hmc, newdata = data.matrix(xgrid), type = "probabilities")
# 
# # predict.svmpath()
# 
# 
# fit_lda <- MASS::lda(as.factor(subs_type) ~ x1 + x2, data = vlm_better_vlm)
# prob_lda <- predict(fit_lda, newdata = xgrid)$posterior
# 
# xgrid2 <- xgrid %>%
#   cbind("LDA" = prob_lda[, 1L]) %>%
#   tidyr::gather(Model, Prob, -x1, -x2)
# 
# 
# vlm_better_vlm %>%
#   ggplot(aes(x1, x2)) +
#   geom_point(aes(color = subs_type, shape = subs_type)) +
#   # scale_color_brewer(palette = "Dark2") +
#   stat_contour(data = xgrid2, aes(x = x1, y = x2, z = Prob, linetype = Model), 
#                breaks = 0.5, color = "black")


