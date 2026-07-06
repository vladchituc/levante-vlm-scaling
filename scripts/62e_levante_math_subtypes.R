# 62e_levante_math_subtypes.R — egma-math broken down by item subtype
#
# SPEC (Vlad, chat 2026-07-06: "can you break the math down by task?"):
#   Variables:   per-trial VLM correctness on egma-math items; subtype
#                (trial_type) + chance_level from the authors' item bank
#                (data/levante-bench/corpus/corpus/egma-math/math-item-bank.csv).
#   Transform:   PRIMARY: self-calibrated theta per subtype (fit_selfcal, JML
#                Rasch, validated in 62d; small-J bias re-quantified below).
#                SECONDARY: child-anchored theta (fit_theta, validated in 62c)
#                for subtypes with >= 10 child-calibrated items.
#   Fitting:     alpha = OLS slope of theta on ln(params_b) per family x subtype
#                (same estimator/space rationale as 62c: additive error on
#                theta = multiplicative on Psi).
#   Aggregation: trials pooled to k/n per model x subtype x item.
#   Level:       family x subtype; headline internvl35 (n=6), qwen35 reported.
#   Exclusions:  subtypes with < 15 administered items (inestimable);
#                floor cells: acc <= subtype chance + 0.10 (chance from bank —
#                replaces 62c's global 0.35 heuristic, which is wrong for the
#                2-option comparison [.50] and slider [.10] subtypes);
#                degenerate persons/items handled inside fit_selfcal.
#   Edge cases:  items in v1 runs missing from the bank -> reported, dropped.
#
# Inputs:  data/levante-bench/v1/*/*/egma-math.csv, corpus math item bank,
#          comparison exports (child d), scripts/62_levante_helpers.R
# Outputs: processed_data/levante_math_subtype_theta.csv
#          processed_data/levante_math_subtype_alpha.csv
#          data/62e_levante_math_subtypes_results.json (seed included)
#          figures/levante_math_subtypes.png

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

MIN_ITEMS <- 15
HEADLINE_FAMILIES <- c("internvl35", "qwen35")

## ---- 0. Small-J validation: quantify self-cal bias at subtype item counts ----
validate <- local({
  n_sim <- 100; J <- 20; n_per <- 10
  sizes <- c(1, 2, 4, 8, 14, 38); a_true <- 0.30; n_extra <- 14
  a_rec <- map_dbl(1:n_sim, function(s) {
    th <- c(-1 + a_true * log(sizes), runif(n_extra, -2, 2))
    d  <- runif(J, -2, 2)
    counts <- expand_grid(m = seq_along(th), i = seq_len(J)) |>
      mutate(model = paste0("m", sprintf("%02d", m)),
             item_uid = paste0("i", i),
             k = rbinom(n(), n_per, plogis(th[m] + d[i])), n = n_per)
    out <- fit_selfcal(counts)
    if (is.null(out)) return(NA_real_)
    fam <- out$theta |> filter(model %in% paste0("m0", 1:6)) |>
      arrange(model) |> mutate(lp = log(sizes))
    coef(lm(theta_self ~ lp, data = fam))[2]
  })
  list(J = J, alpha_bias = mean(a_rec, na.rm = TRUE) - a_true,
       alpha_sd = sd(a_rec, na.rm = TRUE))
})
cat(sprintf("VALIDATION (J=20 self-cal): alpha bias %.4f (sd %.3f)\n",
            validate$alpha_bias, validate$alpha_sd))
stopifnot(abs(validate$alpha_bias) < 0.05)

## ---- 1. Load math trials + item bank ----------------------------------------
task_files <- list.files(file.path(lb, "v1"), pattern = "^egma-math\\.csv$",
                         recursive = TRUE, full.names = TRUE)
trials <- map_dfr(task_files, function(f) {
  p <- parse_v1_path(f)
  if (!p$keep) return(NULL)
  read_csv(f, col_types = cols(.default = col_character()), progress = FALSE) |>
    transmute(model = p$model, item_uid, correct = is_correct == "True")
})

bank <- read_csv(file.path(lb, "corpus", "corpus", "egma-math", "math-item-bank.csv"),
                 col_types = cols(.default = col_character()), progress = FALSE) |>
  filter(!is.na(item_uid), item_uid != "") |>
  distinct(item_uid, trial_type, chance_level) |>
  mutate(chance = as.numeric(chance_level))

counts <- trials |>
  group_by(model, item_uid) |>
  summarise(k = sum(correct), n = n(), .groups = "drop")

n_items_v1 <- n_distinct(counts$item_uid)
unmatched_items <- setdiff(unique(counts$item_uid), bank$item_uid)
counts <- inner_join(counts, bank |> select(item_uid, trial_type, chance),
                     by = "item_uid")
cat(sprintf("Item-bank join: %d of %d v1 math items matched (%d unmatched)\n",
            n_distinct(counts$item_uid), n_items_v1, length(unmatched_items)))
cat("Unmatched item_uid prefixes:\n")
print(table(str_extract(unmatched_items, "^[a-z_]+[a-z]")))

# One chance level per subtype (chance varies by item within a subtype for a
# handful of rows; median is the subtype's dominant response format)
chance_by_type <- counts |>
  group_by(trial_type) |>
  summarise(chance = median(chance, na.rm = TRUE), .groups = "drop")

subtype_sizes <- counts |> distinct(trial_type, item_uid) |> count(trial_type)
print(subtype_sizes)
keep_types <- subtype_sizes |> filter(n >= MIN_ITEMS) |> pull(trial_type)
cat("Subtypes analyzed (>=", MIN_ITEMS, "items):", paste(keep_types, collapse = ", "), "\n")

## ---- 2. Self-calibrated theta per subtype ------------------------------------
fits <- keep_types |> set_names() |>
  map(~ fit_selfcal(counts |> filter(trial_type == .x) |>
                      select(model, item_uid, k, n)))

theta_sub <- imap_dfr(fits, function(f, tt) {
  if (is.null(f)) return(NULL)
  f$theta |> mutate(trial_type = tt)
}) |>
  left_join(counts |> group_by(model, trial_type) |>
              summarise(acc = sum(k) / sum(n), .groups = "drop"),
            by = c("model", "trial_type")) |>
  left_join(chance_by_type, by = "trial_type") |>
  mutate(params_b = unname(levante_params_b[model]),
         family = levante_family(model),
         floor_flag = acc <= chance + 0.10)
write_csv(theta_sub, file.path(root, "processed_data", "levante_math_subtype_theta.csv"))

## ---- 3. Child-anchored theta per subtype (where calibrated items allow) ------
diff_child <- list.files(file.path(lb, "comparison"), pattern = "^egma-math.*_accuracy\\.csv$",
                         full.names = TRUE) |>
  map_dfr(~ read_csv(.x, col_types = "cccdd", progress = FALSE)) |>
  filter(is.finite(difficulty)) |> distinct(item_uid, d_child = difficulty)

anchored <- counts |>
  inner_join(diff_child, by = "item_uid") |>
  group_by(trial_type) |> filter(n_distinct(item_uid) >= 10) |>
  ungroup() |>
  group_by(model, trial_type) |>
  group_modify(~ bind_cols(fit_theta(.x$k, .x$n, .x$d_child),
                           tibble(n_items = nrow(.x), acc = sum(.x$k) / sum(.x$n)))) |>
  ungroup() |>
  mutate(params_b = unname(levante_params_b[model]),
         family = levante_family(model)) |>
  left_join(chance_by_type, by = "trial_type") |>
  mutate(floor_flag = acc <= chance + 0.10)
stopifnot(nrow(anchored) == nrow(distinct(anchored, model, trial_type)))

calib_coverage <- counts |> distinct(trial_type, item_uid) |>
  left_join(diff_child |> mutate(cal = TRUE), by = "item_uid") |>
  group_by(trial_type) |>
  summarise(n_items = n(), n_calibrated = sum(!is.na(cal)), .groups = "drop")

## ---- 4. alpha per family x subtype -------------------------------------------
fit_alpha_col <- function(df, ycol) {
  m <- lm(reformulate("log(params_b)", ycol), data = df)
  ci <- tryCatch(confint(m)[2, ], error = function(e) c(NA, NA))
  tibble(alpha = unname(coef(m)[2]), ci_lo = ci[1], ci_hi = ci[2],
         r2 = summary(m)$r.squared, n_models = nrow(df))
}

alpha_self <- theta_sub |>
  filter(!is.na(params_b), !floor_flag, converged, family %in% HEADLINE_FAMILIES) |>
  group_by(trial_type, family) |> filter(n() >= 3) |>
  group_modify(~ fit_alpha_col(.x, "theta_self")) |> ungroup() |>
  mutate(calibration = "self")

alpha_child <- anchored |>
  filter(!is.na(params_b), !floor_flag, converged, family %in% HEADLINE_FAMILIES) |>
  group_by(trial_type, family) |> filter(n() >= 3) |>
  group_modify(~ fit_alpha_col(.x, "theta")) |> ungroup() |>
  mutate(calibration = "child_anchored")

alpha_sub <- bind_rows(alpha_self, alpha_child)
write_csv(alpha_sub, file.path(root, "processed_data", "levante_math_subtype_alpha.csv"))

## ---- 5. Figure + JSON ----------------------------------------------------------
plot_df <- theta_sub |>
  filter(!is.na(params_b), !floor_flag, family %in% HEADLINE_FAMILIES)
p <- ggplot(plot_df, aes(params_b, theta_self, color = family)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~ trial_type, scales = "free_y") +
  labs(x = "Parameters (billions, nominal, log scale)",
       y = expression(paste("Self-calibrated ability ", theta, " = ln ", Psi)),
       color = "Family",
       title = "EGMA math subtypes: VLM ability vs model size") +
  theme_classic(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), strip.background = element_blank())
ggsave(file.path(root, "figures", "levante_math_subtypes.png"), p,
       width = 10, height = 7, dpi = 200, bg = "white")

write_json(
  list(seed = SEED, validation_small_J = validate,
       n_items_v1 = n_items_v1, n_unmatched_items = length(unmatched_items),
       unmatched_items = unmatched_items,
       subtype_sizes = subtype_sizes,
       calibration_coverage = calib_coverage,
       min_items = MIN_ITEMS,
       floor_rule = "acc <= chance + 0.10 (chance from item bank)",
       alpha = alpha_sub),
  file.path(root, "data", "62e_levante_math_subtypes_results.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 6)

cat("\n== alpha by math subtype ==\n")
print(alpha_sub |> arrange(calibration, trial_type, family), n = 60)
cat("\n== child-calibration coverage by subtype ==\n")
print(calib_coverage)
