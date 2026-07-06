# 62g_levante_compute_axis.R — Training-compute axis for the InternVL3.5 family
#
# SPEC (Vlad, chat 2026-07-06: "can you dig up training compute?"):
#   Resource = LLM-backbone pretraining compute, C = 6 * N_backbone * D:
#     - InternVL3.5 language backbones (SOURCED, InternVL3.5 report arXiv
#       2508.18265 Table 1): 1B->Qwen3-0.6B, 2B->Qwen3-1.7B, 4B->Qwen3-4B,
#       8B->Qwen3-8B, 14B->Qwen3-14B, 38B->Qwen3-32B.
#     - D = 36e12 tokens for ALL Qwen3 sizes (SOURCED, Qwen3 report arXiv
#       2505.09388) — constant across the family.
#     - 6ND: standard dense-transformer FLOP approximation (Kaplan/Chinchilla).
#   EXCLUDED from C (stated approximation): vision-encoder pretraining and the
#   VLM adaptation stages (small relative to 36T-token LLM pretraining; but
#   note the 38B swaps InternViT-300M -> InternViT-6B, so the top point's
#   vision tower differs from the rest).
#   Families NOT fit on this axis (unsourced token counts as of 2026-07-06):
#   qwen35 (Qwen3.5 base pretraining scale unpublished), gemma4, smolvlm2.
#   Capability = theta_self from 62d (self-calibrated, per the presentation
#   decision logged 2026-07-06). alpha_compute = OLS slope of theta_self on
#   ln(C). Because D is constant, alpha_compute also equals the slope on
#   ln(N_backbone); both reported. Floor/converged filters inherited from 62d
#   output file.
#
# Inputs:  processed_data/levante_selfcal_theta.csv (62d), constants above
# Outputs: processed_data/levante_compute_alpha.csv
#          data/62g_levante_compute_results.json
#          figures/levante_compute_axis.png

options(scipen = 999)
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(purrr); library(tidyr)
  library(ggplot2); library(jsonlite); library(tibble)
})

set.seed(42)
SEED <- 42
root <- "."  # repo root; run scripts from the repository root

# Backbone params (billions): InternVL3.5 report Table 1 (arXiv 2508.18265)
backbones <- tribble(
  ~model,            ~backbone,      ~n_backbone_b,
  "internvl35-1B",   "Qwen3-0.6B",   0.6,
  "internvl35-2B",   "Qwen3-1.7B",   1.7,
  "internvl35-4B",   "Qwen3-4B",     4,
  "internvl35-8B",   "Qwen3-8B",     8,
  "internvl35-14B",  "Qwen3-14B",    14,
  "internvl35-38B",  "Qwen3-32B",    32
)
D_TOKENS <- 36e12   # Qwen3 report (arXiv 2505.09388): all sizes, 36T tokens

theta <- read_csv(file.path(root, "processed_data", "levante_selfcal_theta.csv"),
                  show_col_types = FALSE) |>
  inner_join(backbones, by = "model") |>
  mutate(flops = 6 * n_backbone_b * 1e9 * D_TOKENS)
stopifnot(nrow(theta) == 6 * n_distinct(theta$task))

fit_axis <- function(df, xcol) {
  m <- lm(reformulate(sprintf("log(%s)", xcol), "theta_self"), data = df)
  ci <- tryCatch(confint(m)[2, ], error = function(e) c(NA, NA))
  tibble(alpha = unname(coef(m)[2]), ci_lo = ci[1], ci_hi = ci[2],
         r2 = summary(m)$r.squared, n_models = nrow(df))
}

# mental-rotation excluded [Vlad, 2026-07-06]: at 2AFC guessing floor at every
# size, no scaling signal (see 62d).
alpha_compute <- theta |>
  filter(task != "mental-rotation", !floor_flag, converged) |>
  group_by(task) |> filter(n() >= 4) |>
  group_modify(~ bind_rows(
    fit_axis(.x, "flops")        |> mutate(axis = "training_FLOPs_6ND"),
    fit_axis(.x, "params_b")     |> mutate(axis = "nominal_params (62d headline)"))) |>
  ungroup() |>
  mutate(family = "internvl35",
         capability = "psi_self = exp(theta), model-only Rasch calibration")
write_csv(alpha_compute, file.path(root, "processed_data", "levante_compute_alpha.csv"))

p <- ggplot(theta |> filter(task != "mental-rotation", !floor_flag),
            aes(flops, theta_self, color = task)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "LLM pretraining compute, 6ND FLOPs (log scale)",
       y = expression(paste("Self-calibrated ability ", theta, " = ln ", Psi)),
       color = "Task",
       title = "InternVL3.5: ability vs backbone training compute") +
  theme_classic(base_size = 11)
ggsave(file.path(root, "figures", "levante_compute_axis.png"), p,
       width = 7.5, height = 4.5, dpi = 200, bg = "white")

write_json(
  list(seed = SEED,
       sources = list(
         backbones = "InternVL3.5 report arXiv 2508.18265 Table 1",
         tokens = "Qwen3 report arXiv 2505.09388: 36T tokens, all sizes",
         flop_rule = "C = 6*N*D dense approximation"),
       excluded = "vision-encoder + VLM-stage compute; qwen35/gemma4/smolvlm2 (unsourced token counts)",
       flops_range = range(theta$flops),
       alpha = alpha_compute),
  file.path(root, "data", "62g_levante_compute_results.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 6)

cat("== alpha on training-compute axis (internvl35) ==\n")
print(alpha_compute |> arrange(task, axis), n = 30)
