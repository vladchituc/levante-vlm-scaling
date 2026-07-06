# 62b_levante_data_check.R — Pre-flight data check for LEVANTE-bench mirror
#
# Runs BEFORE any analysis (per ~/memory/rules/data-quality.md).
# Input:  data/levante-bench/v1/<model>/<run>/<task>.csv   (trial-level VLM results)
#         data/levante-bench/comparison/<task>_<model>_accuracy.csv (item difficulty map)
# Output: processed_data/62b_levante_data_check.json + printed report
#
# Checks:
#  1. Runs per model x task (expect 10 per metadata num_runs)
#  2. Uniqueness: item_uid unique within model x run x task
#  3. Range: is_correct parses to boolean; difficulty finite
#  4. Consistency: item difficulty identical across comparison files (same item bank)
#  5. Join coverage: share of v1 items with a child-calibrated difficulty
#  6. Sign check: exported `difficulty` should be EASINESS (positively correlated
#     with pooled item accuracy) per repo README "higher d = empirically easier"
#  7. Parser health: share of low/none parse_confidence per model x task

options(scipen = 999)
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(tidyr)
  library(stringr)
  library(jsonlite)
})

root <- "."  # repo root; run scripts from the repository root
lb   <- file.path(root, "data", "levante-bench")
source(file.path(root, "scripts", "62_levante_helpers.R"))

## ---- Load v1 trial CSVs -----------------------------------------------------
task_files <- list.files(file.path(lb, "v1"), pattern = "^(egma-math|matrix-reasoning|mental-rotation|theory-of-mind|trog|vocab)\\.csv$",
                         recursive = TRUE, full.names = TRUE)
cat(length(task_files), "task CSVs found\n")

trials <- map_dfr(task_files, function(f) {
  p <- parse_v1_path(f)
  if (!p$keep) return(NULL)   # flat deterministic-order files: see helpers
  read_csv(f, col_types = cols(.default = col_character()), progress = FALSE) |>
    transmute(model = p$model, run = p$run, task = p$task,
              item_uid = normalize_item_uid(task, item_uid),
              is_correct,
              parse_confidence)
})

## ---- Load comparison difficulty map -----------------------------------------
comp_files <- list.files(file.path(lb, "comparison"), pattern = "_accuracy\\.csv$",
                         full.names = TRUE)
comp <- map_dfr(comp_files, ~ read_csv(.x, col_types = "cccdd", progress = FALSE))

## ---- 1. Runs per model x task ----------------------------------------------
runs_tbl <- trials |> distinct(model, task, run) |> count(model, task, name = "n_runs")
bad_runs <- filter(runs_tbl, n_runs != 10)

## ---- 2. Uniqueness within model x run x task --------------------------------
dups <- trials |> count(model, run, task, item_uid) |> filter(n > 1)

## ---- 3. Range checks ---------------------------------------------------------
ok_bool <- all(trials$is_correct %in% c("True", "False"))
bad_diff <- comp |> filter(!is.finite(difficulty))
na_diff_by_task <- comp |>
  group_by(task) |>
  summarise(n_rows = n(), pct_na_difficulty = round(100 * mean(is.na(difficulty)), 1),
            .groups = "drop")

## ---- 4. Difficulty consistency across comparison files ----------------------
diff_map <- comp |> distinct(task, item_uid, difficulty)
inconsistent <- diff_map |> count(task, item_uid) |> filter(n > 1)

## ---- 5. Join coverage --------------------------------------------------------
v1_items  <- trials |> distinct(task, item_uid)
# comparison task names match v1 task names exactly (egma-math, trog, ...)
unmatched <- anti_join(v1_items, diff_map, by = c("task", "item_uid"))
coverage  <- v1_items |>
  left_join(diff_map |> distinct(task, item_uid) |> mutate(has_d = TRUE),
            by = c("task", "item_uid")) |>
  group_by(task) |>
  summarise(n_items = n(), n_with_d = sum(!is.na(has_d)),
            pct = round(100 * n_with_d / n_items, 1))

## ---- 6. Sign check: difficulty vs pooled accuracy ----------------------------
item_acc <- trials |>
  mutate(correct = is_correct == "True") |>
  group_by(task, item_uid) |>
  summarise(acc = mean(correct), .groups = "drop") |>
  inner_join(diff_map, by = c("task", "item_uid"))
safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  round(cor(x[ok], y[ok]), 3)
}
sign_check <- item_acc |>
  group_by(task) |>
  summarise(r_d_acc = safe_cor(difficulty, acc),
            n_pairs = sum(is.finite(difficulty)), n = n(), .groups = "drop")

## ---- 7. Parser health ---------------------------------------------------------
parse_health <- trials |>
  group_by(model, task) |>
  summarise(pct_low_none = round(100 * mean(parse_confidence %in% c("low", "none")), 1),
            .groups = "drop") |>
  arrange(desc(pct_low_none))

## ---- Report -------------------------------------------------------------------
cat("\n== 1. Runs per model x task: deviations from 10 ==\n")
if (nrow(bad_runs)) print(bad_runs, n = 50) else cat("none\n")
cat("\n== 2. Duplicate item_uid within model x run x task ==\n")
if (nrow(dups)) print(dups, n = 20) else cat("none\n")
cat("\n== 3. is_correct strictly True/False:", ok_bool,
    "| non-finite difficulties:", nrow(bad_diff), "==\n")
cat("\n== 3b. NA difficulty share by task (comparison files) ==\n")
print(na_diff_by_task)
cat("\n== 4. Items with >1 distinct difficulty across files ==\n")
if (nrow(inconsistent)) print(inconsistent, n = 20) else cat("none\n")
cat("\n== 5. Difficulty coverage of v1 items ==\n"); print(coverage)
cat("\n== 6. Sign check (expect POSITIVE r if d = easiness) ==\n"); print(sign_check)
cat("\n== 7. Parser health: worst 10 model x task ==\n"); print(head(parse_health, 10))

write_json(
  list(n_task_files = length(task_files),
       n_trials = nrow(trials),
       n_models = n_distinct(trials$model),
       runs_deviating_from_10 = bad_runs,
       n_duplicate_item_rows = nrow(dups),
       is_correct_boolean = ok_bool,
       n_nonfinite_difficulty = nrow(bad_diff),
       na_difficulty_by_task = na_diff_by_task,
       n_items_inconsistent_difficulty = nrow(inconsistent),
       coverage = coverage,
       sign_check = sign_check,
       n_unmatched_items = nrow(unmatched),
       unmatched_by_task = count(unmatched, task),
       parse_health_worst = head(parse_health, 20)),
  file.path(root, "processed_data", "62b_levante_data_check.json"),
  auto_unbox = TRUE, pretty = TRUE)
cat("\nWrote processed_data/62b_levante_data_check.json\n")
