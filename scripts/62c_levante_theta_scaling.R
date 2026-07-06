# 62c_levante_theta_scaling.R — VLM scaling exponents on child-anchored Rasch ability
#
# CONFIRMED SPEC (Vlad, chat 2026-07-06: proposed design approved "start with the models"):
#   Variables:   per-trial VLM correctness (is_correct) on LEVANTE items;
#                child-calibrated Rasch item easiness d (comparison exports).
#   Transform:   theta per model x task by ML with item parameters FIXED at child
#                values: P(correct) = plogis(theta + d)  [mirt easiness form;
#                sign verified in 62b: r(d, acc) > 0 all usable tasks].
#                Under the log-interval reading theta = ln(Psi), so
#                Psi = e^theta is the ratio-scale capability. A power law
#                Psi = a * R^alpha  <=>  theta = ln(a) + alpha * ln(R):
#                alpha is estimated DIRECTLY as the slope of theta on ln(params).
#   Fitting:     OLS, theta ~ ln(params). Error structure: theta is a logit-scale
#                ML estimate, approx normal — additive error on theta =
#                multiplicative on Psi, so OLS in log space is the correct choice
#                (consistent with catalog protocol step 3).
#   Aggregation: trials pooled across runs within model x task x item (k of n);
#                one theta per model x task; fits per family x task.
#   Level:       reported per family x task (headline: internvl35 n=6,
#                qwen35 n=5, smolvlm2 n=3; sensitivity: gemma4-it [mixed
#                effective/total params], tinyllava n=2 recorded without CI).
#   Exclusions:  closed models (unsourced params), gemma4-26B-A4B (MoE),
#                molmo2 (single size); tasks without child calibration
#                (theory-of-mind, mental-rotation, matrix-reasoning);
#                degenerate cells (all-0 or all-1); floor-flagged cells
#                (pooled acc <= 0.35, HEURISTIC near-chance zone) kept in the
#                theta table but excluded from headline fits (sensitivity: kept).
#   Edge cases:  glm may not converge at extreme theta -> tryCatch, flag.
#   Assumption:  plain Rasch, NO guessing floor (authors' calibration used
#                guessing bounds; floors not exported). Floor-flag exclusion is
#                the mitigation. Revisit when Redivis .rds models are available.
#   Multiplicity: estimation only (alpha + CI), no NHST grid -> no adjustment.
#
# Inputs:  data/levante-bench/v1/<model>/<run>/{trog,vocab,egma-math}.csv
#          data/levante-bench/comparison/*_accuracy.csv
#          scripts/62_levante_helpers.R
# Outputs: processed_data/levante_theta_by_model.csv
#          processed_data/levante_alpha_catalog.csv
#          data/62c_levante_results.json (includes seed + validation)
#          figures/levante_theta_scaling.png

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

USABLE_TASKS <- c("trog", "vocab", "egma-math")
FLOOR_ACC    <- 0.35   # heuristic near-chance zone for 3-4 option tasks
HEADLINE_FAMILIES <- c("internvl35", "qwen35", "smolvlm2")

## =============================================================================
## 0. VALIDATION GATE — recover known theta and alpha before touching real data
##    (required per r-research.md; simulated Rasch data, same estimator code)
## =============================================================================

# fit_theta() lives in 62_levante_helpers.R (moved 2026-07-06; validated here)

validate <- local({
  n_sim <- 200; J <- 100; n_per <- 10
  # theta recovery across a realistic grid
  th_true <- runif(n_sim, -2, 3)
  th_rec <- map_dbl(th_true, function(th) {
    d <- runif(J, -3, 4)
    k <- rbinom(J, n_per, plogis(th + d))
    fit_theta(k, rep(n_per, J), d)$theta
  })
  theta_bias <- mean(th_rec - th_true)
  theta_rmse <- sqrt(mean((th_rec - th_true)^2))
  # alpha recovery through the full pipeline (6 sizes, alpha = 0.30)
  sizes <- c(1, 2, 4, 8, 14, 38); a_true <- 0.30
  a_rec <- map_dbl(1:n_sim, function(i) {
    th <- -1 + a_true * log(sizes)
    thetas <- map_dbl(th, function(t) {
      d <- runif(J, -3, 4)
      k <- rbinom(J, n_per, plogis(t + d))
      fit_theta(k, rep(n_per, J), d)$theta
    })
    coef(lm(thetas ~ log(sizes)))[2]
  })
  alpha_bias <- mean(a_rec - a_true)
  list(theta_bias = theta_bias, theta_rmse = theta_rmse,
       alpha_bias = alpha_bias, alpha_sd = sd(a_rec))
})
cat(sprintf("VALIDATION: theta bias %.4f (rmse %.3f), alpha bias %.4f (sd %.3f)\n",
            validate$theta_bias, validate$theta_rmse,
            validate$alpha_bias, validate$alpha_sd))
stopifnot(abs(validate$theta_bias) < 0.05, abs(validate$alpha_bias) < 0.02)

## =============================================================================
## 1. Load data
## =============================================================================

task_re <- paste0("^(", paste(USABLE_TASKS, collapse = "|"), ")\\.csv$")
task_files <- list.files(file.path(lb, "v1"), pattern = task_re,
                         recursive = TRUE, full.names = TRUE)
trials <- map_dfr(task_files, function(f) {
  p <- parse_v1_path(f)
  if (!p$keep) return(NULL)   # flat deterministic-order files: see helpers
  read_csv(f, col_types = cols(.default = col_character()), progress = FALSE) |>
    transmute(model = p$model, task = p$task,
              item_uid = normalize_item_uid(task, item_uid),
              correct = is_correct == "True")
})

diff_map <- list.files(file.path(lb, "comparison"), pattern = "_accuracy\\.csv$",
                       full.names = TRUE) |>
  map_dfr(~ read_csv(.x, col_types = "cccdd", progress = FALSE)) |>
  filter(task %in% USABLE_TASKS, is.finite(difficulty)) |>
  distinct(task, item_uid, difficulty)
stopifnot(nrow(count(diff_map, task, item_uid) |> filter(n > 1)) == 0)

## =============================================================================
## 2. Pool to k/n per model x task x item; join child difficulties
## =============================================================================

counts <- trials |>
  group_by(model, task, item_uid) |>
  summarise(k = sum(correct), n = n(), .groups = "drop")

n_before <- nrow(counts)
counts <- inner_join(counts, diff_map, by = c("task", "item_uid"))
cat(sprintf("Calibrated-item join: kept %d of %d model x task x item cells (%.1f%%)\n",
            nrow(counts), n_before, 100 * nrow(counts) / n_before))

## =============================================================================
## 3. Theta per model x task (child-anchored ML)
## =============================================================================

theta_tbl <- counts |>
  group_by(model, task) |>
  group_modify(function(df, key) {
    bind_cols(fit_theta(df$k, df$n, df$difficulty),
              tibble(n_items = nrow(df),
                     n_trials = sum(df$n),
                     acc = sum(df$k) / sum(df$n)))
  }) |>
  ungroup() |>
  mutate(psi = exp(theta),
         degenerate = acc %in% c(0, 1),
         floor_flag = acc <= FLOOR_ACC,
         params_b = unname(levante_params_b[model]),
         family = levante_family(model))

write_csv(theta_tbl, file.path(root, "processed_data", "levante_theta_by_model.csv"))

## =============================================================================
## 4. Power-law fits: alpha = slope of theta on ln(params), per family x task
## =============================================================================

fit_alpha <- function(df, n_boot = 2000) {
  m <- lm(theta ~ log(params_b), data = df)
  ci <- tryCatch(confint(m)[2, ], error = function(e) c(NA, NA))
  boot <- if (nrow(df) >= 4) {
    reps <- map_dbl(1:n_boot, function(i) {
      idx <- sample(nrow(df), replace = TRUE)
      if (n_distinct(df$params_b[idx]) < 2) return(NA_real_)
      coef(lm(theta ~ log(params_b), data = df[idx, ]))[2]
    })
    quantile(reps, c(.025, .975), na.rm = TRUE)
  } else c(NA_real_, NA_real_)
  tibble(alpha = unname(coef(m)[2]),
         ci_lo = ci[1], ci_hi = ci[2],
         boot_lo = boot[1], boot_hi = boot[2],
         r2 = summary(m)$r.squared, n_models = nrow(df))
}

fit_input <- theta_tbl |>
  filter(!is.na(params_b), !degenerate, !floor_flag, converged)

alpha_cat <- fit_input |>
  group_by(task, family) |>
  filter(n() >= 2) |>
  group_modify(~ fit_alpha(.x)) |>
  ungroup() |>
  mutate(headline = family %in% HEADLINE_FAMILIES,
         currency = "params_nominal",
         capability = "psi = exp(theta), child-anchored Rasch")

# Sensitivity A: floor-flagged cells kept
alpha_sens_floor <- theta_tbl |>
  filter(!is.na(params_b), !degenerate, converged) |>
  group_by(task, family) |> filter(n() >= 2) |>
  group_modify(~ fit_alpha(.x)) |> ungroup() |>
  mutate(sensitivity = "floor_cells_included")

# Sensitivity B: naive pooled-accuracy odds on the same items (protocol's
# benchmark transform, no IRT anchoring)
alpha_sens_odds <- theta_tbl |>
  filter(!is.na(params_b), !degenerate, !floor_flag, converged) |>
  mutate(theta = log(acc / (1 - acc))) |>   # reuse fit_alpha via theta column
  group_by(task, family) |> filter(n() >= 2) |>
  group_modify(~ fit_alpha(.x)) |> ungroup() |>
  mutate(sensitivity = "pooled_accuracy_odds")

write_csv(alpha_cat,        file.path(root, "processed_data", "levante_alpha_catalog.csv"))
write_csv(bind_rows(alpha_sens_floor, alpha_sens_odds),
          file.path(root, "processed_data", "levante_alpha_sensitivities.csv"))

## =============================================================================
## 5. Figure
## =============================================================================

plot_df <- fit_input |> filter(family %in% HEADLINE_FAMILIES |
                                 family %in% c("gemma4", "tinyllava"))
p <- ggplot(plot_df, aes(params_b, theta, color = family)) +
  geom_point(size = 2) +
  geom_smooth(data = filter(plot_df, family %in% HEADLINE_FAMILIES),
              method = "lm", se = FALSE, linewidth = 0.6) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~ task, scales = "free_y") +
  labs(x = "Parameters (billions, nominal, log scale)",
       y = expression(paste("Child-anchored ability ", theta, " = ln ", Psi)),
       color = "Family",
       title = "VLM ability on LEVANTE tasks vs model size",
       subtitle = expression(paste("Slope of ", theta, " on ln(params) = power-law exponent ", alpha, " for ", Psi %prop% R^alpha))) +
  theme_classic(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), strip.background = element_blank())
ggsave(file.path(root, "figures", "levante_theta_scaling.png"), p,
       width = 10, height = 4, dpi = 200, bg = "white")

## =============================================================================
## 6. Results JSON
## =============================================================================

write_json(
  list(seed = SEED,
       validation = validate,
       spec = list(model = "P = plogis(theta + d), d fixed at child calibration",
                   alpha = "slope of theta on ln(params_b), OLS",
                   guessing = "none (floor-flag heuristic instead)",
                   floor_acc_threshold = FLOOR_ACC,
                   tasks = USABLE_TASKS),
       n_trials = nrow(trials),
       n_cells_calibrated = nrow(counts),
       theta_cells = nrow(theta_tbl),
       cells_floor_flagged = sum(theta_tbl$floor_flag),
       cells_degenerate = sum(theta_tbl$degenerate),
       alpha_catalog = alpha_cat,
       sensitivities = bind_rows(alpha_sens_floor, alpha_sens_odds)),
  file.path(root, "data", "62c_levante_results.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 6)
cat("Wrote catalog, sensitivities, figure, and data/62c_levante_results.json\n")
print(alpha_cat |> arrange(task, desc(headline), family), n = 40)
