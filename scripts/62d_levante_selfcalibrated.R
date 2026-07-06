# 62d_levante_selfcalibrated.R — Self-calibrated (model-only) Rasch scaling
#
# SPEC (Vlad, chat 2026-07-06: "can we just model their performance on a ratio
# scale independently of what kids do?"):
#   Variables:   same per-trial VLM correctness as 62c, ALL SIX tasks (no child
#                calibration required). Persons = models; items = task items.
#   Transform:   joint Rasch fit per task via two-way binomial logit fixed
#                effects: glm(cbind(k, n-k) ~ model + item, binomial) [JML].
#                Identification: mean item easiness = 0 per task; theta_m =
#                intercept + model effect + mean(item effects). Psi = e^theta.
#   Fitting:     alpha = OLS slope of theta_self on ln(params_b), per family x
#                task (same as 62c). Log-space OLS: additive error on theta =
#                multiplicative on Psi.
#   Aggregation: trials pooled across runs to k/n per model x task x item.
#   Level:       per family x task; headline families internvl35, qwen35.
#   Exclusions:  degenerate persons (acc 0 or 1 within task) and degenerate
#                items (pooled all-0 or all-1) — inestimable fixed effects.
#                ALL models (incl. closed) enter the calibration; scaling fits
#                still restricted to families with sourced nominal params and
#                non-floor cells (same FLOOR_ACC rule as 62c, applied to
#                calibrated-item accuracy).
#   Edge cases:  JML person-parameter bias (incidental parameters) — quantified
#                in the validation gate on simulated data before real fits.
#   Comparison:  child-derived vs self-derived item difficulty per task
#                (r + OLS slope): the slope is the relative unit stretch between
#                the child scale and the model scale, and predicts
#                alpha_child ~= stretch * alpha_self.
#
# Inputs:  data/levante-bench/v1/..., data/levante-bench/comparison/...,
#          scripts/62_levante_helpers.R
# Outputs: processed_data/levante_selfcal_theta.csv
#          processed_data/levante_selfcal_alpha.csv
#          processed_data/levante_scale_comparison.csv
#          data/62d_levante_selfcal_results.json (seed included)
#          figures/levante_selfcal_scaling.png

options(scipen = 999)
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(purrr); library(tidyr)
  library(stringr); library(ggplot2); library(jsonlite); library(tibble)
})

set.seed(42)
SEED <- 42
root <- "."  # repo root; run scripts from the repository root
lb   <- file.path(root, "data", "levante-bench")
source(file.path(root, "scripts", "62_levante_helpers.R"))

ALL_TASKS <- c("trog", "vocab", "egma-math", "matrix-reasoning",
               "mental-rotation", "theory-of-mind")
# Excluded from alpha fits + figures [Vlad, 2026-07-06]: all models sit at the
# 2AFC guessing floor at every size (62d alpha ~ 0.02, R^2 0.07) — no scaling
# signal to fit. theta still estimated and written to the theta CSV.
EXCLUDED_TASKS <- "mental-rotation"
FLOOR_ACC <- 0.35
HEADLINE_FAMILIES <- c("internvl35", "qwen35")

## =============================================================================
## 0. Self-calibrated Rasch fit (JML via two-way logit FE) + VALIDATION GATE
## =============================================================================

# fit_selfcal() lives in 62_levante_helpers.R (moved 2026-07-06; validated here)

validate <- local({
  n_sim <- 100; J <- 100; n_per <- 10
  sizes <- c(1, 2, 4, 8, 14, 38); a_true <- 0.30; n_extra <- 14
  a_rec <- map_dbl(1:n_sim, function(s) {
    th <- c(-1 + a_true * log(sizes), runif(n_extra, -2, 2))  # family + other models
    d  <- runif(J, -2, 2)
    counts <- expand_grid(m = seq_along(th), i = seq_len(J)) |>
      mutate(model = paste0("m", sprintf("%02d", m)),
             item_uid = paste0("i", i),
             k = rbinom(n(), n_per, plogis(th[m] + d[i])), n = n_per)
    out <- fit_selfcal(counts)
    fam <- out$theta |> filter(model %in% paste0("m0", 1:6)) |>
      arrange(model) |> mutate(lp = log(sizes))
    coef(lm(theta_self ~ lp, data = fam))[2]
  })
  list(alpha_bias = mean(a_rec) - a_true, alpha_sd = sd(a_rec))
})
cat(sprintf("VALIDATION (self-cal JML pipeline): alpha bias %.4f (sd %.3f)\n",
            validate$alpha_bias, validate$alpha_sd))
stopifnot(abs(validate$alpha_bias) < 0.03)

## =============================================================================
## 1. Load trials (all six tasks), pool to k/n
## =============================================================================

task_re <- paste0("^(", paste(ALL_TASKS, collapse = "|"), ")\\.csv$")
task_files <- list.files(file.path(lb, "v1"), pattern = task_re,
                         recursive = TRUE, full.names = TRUE)
trials <- map_dfr(task_files, function(f) {
  p <- parse_v1_path(f)
  if (!p$keep) return(NULL)
  read_csv(f, col_types = cols(.default = col_character()), progress = FALSE) |>
    transmute(model = p$model, task = p$task,
              item_uid = normalize_item_uid(task, item_uid),
              correct = is_correct == "True")
})
counts <- trials |>
  group_by(model, task, item_uid) |>
  summarise(k = sum(correct), n = n(), .groups = "drop")

## =============================================================================
## 2. Self-calibrated theta per task
## =============================================================================

fits <- ALL_TASKS |> set_names() |>
  map(~ fit_selfcal(counts |> filter(task == .x) |> select(model, item_uid, k, n)))

theta_self <- imap_dfr(fits, function(f, tk) {
  if (is.null(f)) return(NULL)
  f$theta |> mutate(task = tk)
}) |>
  left_join(counts |> group_by(model, task) |>
              summarise(acc = sum(k) / sum(n), .groups = "drop"),
            by = c("model", "task")) |>
  mutate(psi_self = exp(theta_self),
         params_b = unname(levante_params_b[model]),
         family = levante_family(model),
         floor_flag = acc <= FLOOR_ACC)
write_csv(theta_self, file.path(root, "processed_data", "levante_selfcal_theta.csv"))

## =============================================================================
## 3. alpha per family x task (same estimator as 62c)
## =============================================================================

fit_alpha <- function(df) {
  m <- lm(theta_self ~ log(params_b), data = df)
  ci <- tryCatch(confint(m)[2, ], error = function(e) c(NA, NA))
  tibble(alpha_self = unname(coef(m)[2]), ci_lo = ci[1], ci_hi = ci[2],
         r2 = summary(m)$r.squared, n_models = nrow(df))
}

alpha_self <- theta_self |>
  filter(!task %in% EXCLUDED_TASKS,
         !is.na(params_b), !floor_flag, converged,
         family %in% c(HEADLINE_FAMILIES, "gemma4", "smolvlm2", "tinyllava")) |>
  group_by(task, family) |> filter(n() >= 3) |>
  group_modify(~ fit_alpha(.x)) |> ungroup() |>
  mutate(headline = family %in% HEADLINE_FAMILIES,
         capability = "psi_self = exp(theta), model-only Rasch calibration")

# Pooled-across-families fit (all param-known, non-floor models in one OLS):
# architecture/recipe differences land in the residuals; slope confounds
# within-family scaling with between-family level differences.
alpha_pooled <- theta_self |>
  filter(!task %in% EXCLUDED_TASKS,
         !is.na(params_b), !floor_flag, converged) |>
  group_by(task) |> filter(n() >= 4) |>
  group_modify(~ fit_alpha(.x)) |> ungroup() |>
  mutate(family = "ALL (pooled)", headline = FALSE,
         capability = "psi_self = exp(theta), model-only Rasch calibration")
alpha_self <- bind_rows(alpha_self, alpha_pooled)
write_csv(alpha_self, file.path(root, "processed_data", "levante_selfcal_alpha.csv"))

## =============================================================================
## 4. Scale comparison: child-derived vs self-derived item difficulty
## =============================================================================

diff_child <- list.files(file.path(lb, "comparison"), pattern = "_accuracy\\.csv$",
                         full.names = TRUE) |>
  map_dfr(~ read_csv(.x, col_types = "cccdd", progress = FALSE)) |>
  filter(is.finite(difficulty)) |>
  distinct(task, item_uid, d_child = difficulty)

d_self_all <- imap_dfr(fits, function(f, tk) {
  if (is.null(f)) return(NULL)
  f$d |> mutate(task = tk)
})

scale_cmp <- inner_join(d_self_all, diff_child, by = c("task", "item_uid")) |>
  group_by(task) |>
  summarise(n_items = n(),
            r = cor(d_self, d_child),
            stretch = coef(lm(d_child ~ d_self))[2],   # child units per self unit
            stretch_se = summary(lm(d_child ~ d_self))$coefficients[2, 2],
            .groups = "drop")
write_csv(scale_cmp, file.path(root, "processed_data", "levante_scale_comparison.csv"))

## =============================================================================
## 5. Figure + JSON
## =============================================================================

plot_df <- theta_self |>
  filter(!task %in% EXCLUDED_TASKS,
         !is.na(params_b), !floor_flag, family %in% HEADLINE_FAMILIES)
p <- ggplot(plot_df, aes(params_b, theta_self, color = family)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~ task, scales = "free_y") +
  labs(x = "Parameters (billions, nominal, log scale)",
       y = expression(paste("Self-calibrated ability ", theta, " = ln ", Psi)),
       color = "Family",
       title = "VLM ability on LEVANTE tasks, model-only Rasch calibration") +
  theme_classic(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), strip.background = element_blank())
ggsave(file.path(root, "figures", "levante_selfcal_scaling.png"), p,
       width = 10, height = 6, dpi = 200, bg = "white")

write_json(
  list(seed = SEED, validation = validate,
       spec = list(model = "JML Rasch: glm(cbind(k,n-k) ~ model + item), mean item easiness = 0",
                   alpha = "slope of theta_self on ln(params_b), OLS",
                   floor_acc_threshold = FLOOR_ACC),
       dropped = imap(fits, ~ if (is.null(.x)) NULL else
         list(models = .x$n_dropped_models, items = .x$n_dropped_items)),
       alpha_self = alpha_self,
       scale_comparison = scale_cmp),
  file.path(root, "data", "62d_levante_selfcal_results.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 6)

cat("\n== alpha (self-calibrated) ==\n")
print(alpha_self |> arrange(task, desc(headline), family), n = 40)
cat("\n== scale comparison: child vs self item difficulty ==\n")
print(scale_cmp)
