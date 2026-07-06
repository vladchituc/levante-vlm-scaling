# 62f_levante_aquila_training.R — Training-data exponent from Aquila-VL-2B checkpoints
#
# SPEC (Vlad, chat 2026-07-06: "is there a way to get exponents for not model
# size?"):
#   One architecture (Aquila-VL-2B), five points along its own training run:
#   checkpoints stage2a, stage2b, stage2c, stage3, production. Capability =
#   child-anchored theta (fit_theta, validated 62c) from the authors' public
#   comparison exports (item-level correct 0/1, single run, + child Rasch d).
#   Resource = CUMULATIVE TRAINING SAMPLES (millions), sourced from the
#   Infinity-MM paper (arXiv 2410.18558, Table 2):
#     stage 1 = 10M (projector alignment), stage 2 = 8.2M in three subsets,
#     stage 3 = 8.2M, stage 4 = 6M (production = after stage 4).
#   ASSUMPTIONS (flagged, not sourced): stage 2's 8.2M is split EVENLY across
#   the a/b/c checkpoints (paper gives three subsets, not their sizes);
#   "production" = post-stage-4 release model.
#   Primary axis: cumulative samples INCLUDING stage 1 (all data seen).
#   Sensitivity: instruction samples only (excluding stage-1 alignment).
#   alpha = OLS slope of theta on ln(cumulative samples), per task; n = 5
#   points — REPORT WITH THAT CAVEAT. Tasks: trog, vocab, egma-math (the
#   child-calibrated exports). Floor rule as 62c (acc <= 0.35 excluded).
#
# Inputs:  data/levante-bench/comparison/<task>_aquila_vl_checkpoint_*.csv +
#          <task>_aquila_vl_production_accuracy.csv, 62_levante_helpers.R
# Outputs: processed_data/levante_aquila_training.csv
#          data/62f_levante_aquila_results.json (seed included)
#          figures/levante_aquila_training.png

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
FLOOR_ACC <- 0.35

# Cumulative training samples (millions) at each released checkpoint.
# Source: Infinity-MM (arXiv 2410.18558) Table 2 — stage1 10M, stage2 8.2M
# (three subsets, EVEN SPLIT ASSUMED), stage3 8.2M, stage4 6M.
stages <- tribble(
  ~ckpt,                          ~cum_all_m, ~cum_instr_m,
  "aquila_vl_checkpoint_stage2a", 10 + 8.2/3,       8.2/3,
  "aquila_vl_checkpoint_stage2b", 10 + 2*8.2/3,   2*8.2/3,
  "aquila_vl_checkpoint_stage2c", 10 + 8.2,         8.2,
  "aquila_vl_checkpoint_stage3",  10 + 16.4,       16.4,
  "aquila_vl_production",         10 + 22.4,       22.4
)

## ---- Load comparison exports for the Aquila series ---------------------------
files <- list.files(file.path(lb, "comparison"),
                    pattern = "aquila_vl.*_accuracy\\.csv$", full.names = TRUE)
aq <- map_dfr(files, ~ read_csv(.x, col_types = "cccdd", progress = FALSE)) |>
  filter(task %in% USABLE_TASKS, is.finite(difficulty)) |>
  inner_join(stages, by = c("model" = "ckpt"))
stopifnot(n_distinct(aq$model) == 5)

## ---- Child-anchored theta per checkpoint x task ------------------------------
theta_aq <- aq |>
  group_by(model, task, cum_all_m, cum_instr_m) |>
  group_modify(~ bind_cols(fit_theta(.x$correct, rep(1, nrow(.x)), .x$difficulty),
                           tibble(n_items = nrow(.x), acc = mean(.x$correct)))) |>
  ungroup() |>
  mutate(psi = exp(theta), floor_flag = acc <= FLOOR_ACC)
write_csv(theta_aq, file.path(root, "processed_data", "levante_aquila_training.csv"))

## ---- alpha per task: theta vs ln(cumulative samples) --------------------------
fit_axis <- function(df, xcol) {
  m <- lm(reformulate(sprintf("log(%s)", xcol), "theta"), data = df)
  ci <- tryCatch(confint(m)[2, ], error = function(e) c(NA, NA))
  tibble(alpha = unname(coef(m)[2]), ci_lo = ci[1], ci_hi = ci[2],
         r2 = summary(m)$r.squared, n_points = nrow(df))
}

alpha_aq <- theta_aq |>
  filter(!floor_flag, converged) |>
  group_by(task) |> filter(n() >= 4) |>
  group_modify(~ bind_rows(
    fit_axis(.x, "cum_all_m")   |> mutate(axis = "cumulative_samples_all"),
    fit_axis(.x, "cum_instr_m") |> mutate(axis = "instruction_samples_only"))) |>
  ungroup() |>
  mutate(currency = "training_samples_M",
         capability = "psi = exp(theta), child-anchored Rasch",
         system = "Aquila-VL-2B (one architecture, its own training run)")

## ---- Figure + JSON -------------------------------------------------------------
p <- ggplot(theta_aq |> filter(!floor_flag),
            aes(cum_all_m, theta, color = task)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "Cumulative training samples (millions, log scale)",
       y = expression(paste("Child-anchored ability ", theta, " = ln ", Psi)),
       color = "Task",
       title = "Aquila-VL-2B: ability vs amount of training (fixed size)") +
  theme_classic(base_size = 11)
ggsave(file.path(root, "figures", "levante_aquila_training.png"), p,
       width = 7, height = 4.5, dpi = 200, bg = "white")

write_json(
  list(seed = SEED,
       source = "Infinity-MM arXiv 2410.18558 Table 2; even stage-2 split ASSUMED",
       stages = stages, alpha = alpha_aq,
       floor_flagged = theta_aq |> filter(floor_flag) |> select(model, task, acc)),
  file.path(root, "data", "62f_levante_aquila_results.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 6)

cat("== Aquila training-data exponents ==\n")
print(alpha_aq |> arrange(axis, task))
cat("\n== theta by checkpoint ==\n")
print(theta_aq |> arrange(task, cum_all_m) |>
        select(task, model, cum_all_m, acc, theta, theta_se, floor_flag), n = 20)
