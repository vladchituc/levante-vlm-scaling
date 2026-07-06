# 62_levante_helpers.R — shared helpers for the LEVANTE-bench scripts (62b, 62c)
# Sourced, not run standalone.

# Normalize v1 item_uids to the comparison-export naming so the child-calibrated
# difficulty map joins. Verified mappings (2026-07-06, data check 62b):
#   vocab:  v1 "vocab__acorn"  -> comparison "vocab_word_acorn"
#   others: identical naming where calibrated (trog, egma-math, matrix_ subset)
normalize_item_uid <- function(task, item_uid) {
  ifelse(task == "vocab",
         sub("^vocab__", "vocab_word_", item_uid),
         item_uid)
}

# Nominal parameter counts (billions) by v1 model folder name.
# Source: parameter counts as encoded in the authors' model names / HF model ids
# (nominal totals). NA = excluded from scaling fits (closed models: unsourced;
# gemma4-26B-A4B: MoE, active!=total, mixed currency; molmo2: single-size family).
levante_params_b <- c(
  "internvl35-1B"  = 1,    "internvl35-2B" = 2,   "internvl35-4B" = 4,
  "internvl35-8B"  = 8,    "internvl35-14B" = 14, "internvl35-38B" = 38,
  "qwen35-0.8B"    = 0.8,  "qwen35-2B"     = 2,   "qwen35-4B"     = 4,
  "qwen35-9B"      = 9,    "qwen35-27B"    = 27,
  "smolvlm2-256M"  = 0.256, "smolvlm2-500M" = 0.5, "smolvlm2-2.2B" = 2.2,
  "gemma4-E2B-it"  = 2,    "gemma4-E4B-it" = 4,   "gemma4-31B-it" = 31,
  "gemma4-31B"     = 31,
  "tinyllava-2.4B" = 2.4,  "tinyllava-3.1B" = 3.1
)

levante_family <- function(model) sub("-[^-]*$", "", sub("-it$", "", model))

# --- Validated estimators (validation gates live in 62c / 62d; moved here
# --- 2026-07-06 for reuse by 62e; reruns confirmed identical outputs) --------

# ML theta with item easiness fixed at external calibration: P = plogis(theta + d)
fit_theta <- function(k, n, d) {
  fit <- tryCatch(
    glm(cbind(k, n - k) ~ 1 + offset(d), family = binomial()),
    error = function(e) NULL, warning = function(w) NULL)
  if (is.null(fit)) {
    fit <- suppressWarnings(glm(cbind(k, n - k) ~ 1 + offset(d), family = binomial()))
    converged <- FALSE
  } else converged <- fit$converged
  tibble::tibble(theta = unname(coef(fit)[1]),
                 theta_se = summary(fit)$coefficients[1, 2],
                 converged = converged)
}

# Self-calibrated JML Rasch via two-way binomial logit FE; persons = models.
# counts: model, item_uid, k, n. Identification: mean item easiness = 0.
fit_selfcal <- function(counts, min_models = 3, min_items = 10) {
  acc_m <- counts |> dplyr::group_by(model) |> dplyr::summarise(a = sum(k) / sum(n))
  acc_i <- counts |> dplyr::group_by(item_uid) |> dplyr::summarise(a = sum(k) / sum(n))
  keep_m <- acc_m$model[acc_m$a > 0 & acc_m$a < 1]
  keep_i <- acc_i$item_uid[acc_i$a > 0 & acc_i$a < 1]
  df <- counts |> dplyr::filter(model %in% keep_m, item_uid %in% keep_i) |>
    dplyr::mutate(model = factor(model), item_uid = factor(item_uid))
  if (dplyr::n_distinct(df$model) < min_models ||
      dplyr::n_distinct(df$item_uid) < min_items) return(NULL)
  fit <- glm(cbind(k, n - k) ~ model + item_uid, family = binomial(), data = df)
  cf <- coef(fit)
  m_lev <- levels(df$model); i_lev <- levels(df$item_uid)
  m_eff <- c(0, cf[paste0("model", m_lev[-1])])
  i_eff <- c(0, cf[paste0("item_uid", i_lev[-1])])
  theta <- unname(cf["(Intercept)"] + m_eff + mean(i_eff))
  d_self <- unname(i_eff - mean(i_eff))
  list(theta = tibble::tibble(model = m_lev, theta_self = theta,
                              converged = fit$converged),
       d = tibble::tibble(item_uid = i_lev, d_self = d_self),
       n_dropped_models = sum(!acc_m$model %in% keep_m),
       n_dropped_items = sum(!acc_i$item_uid %in% keep_i))
}

# Parse a mirrored v1 file path into (model, run, task). Layouts in the bucket:
#   v1/<model>/<run>/<task>.csv   run = "0001".."0010" or "baseline"  -> keep
#   v1/<model>/<task>.csv         flat deterministic-order extra run  -> DROP
#     (paper protocol = randomized-order runs; flat files are a different
#      option-order protocol and would unbalance per-item trial counts)
parse_v1_path <- function(f) {
  parts <- strsplit(f, "/")[[1]]
  n <- length(parts)
  i_v1 <- max(which(parts == "v1"))
  depth <- n - i_v1        # 3 = model/run/task.csv ; 2 = model/task.csv (flat)
  if (depth == 3) {
    list(model = parts[i_v1 + 1], run = parts[i_v1 + 2],
         task = sub("\\.csv$", "", parts[n]), keep = TRUE)
  } else {
    list(model = parts[i_v1 + 1], run = "flat",
         task = sub("\\.csv$", "", parts[n]), keep = FALSE)
  }
}
