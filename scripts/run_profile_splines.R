#!/usr/bin/env Rscript

# Fit matched thin-plate spline reconstructions to the exact bootstrap design
# matrices exported by prepare_spline_profile_inputs.py.  This runner keeps the
# bootstrap samples, profile outcomes, map aggregation, and recovery metrics
# fixed so that the fitted spatial regularizer is the method being compared.

SCRIPT_VERSION <- "profile-spline-runner-v1"
INPUT_FORMAT_VERSION <- "spline_profile_inputs_v1"
SWEEP_DESIGN_VERSION <- "factorial_nested_prefix_median_v1"
SEED_POLICY_VERSION <- "sha256_slot_seed_v1"
NX <- 20L
NY <- 10L
N_CELLS <- NX * NY
BASIS_K <- as.integer(N_CELLS / 10L)
BASIS_ID <- "tp-k20-xmajor-yfast-absorbcons-false-nointercept-v1"
DEFAULT_GAM_METHOD <- "GCV.Cp"
MAX_PSOCK_WORKERS <- 4L
SUPPORT_TOLERANCE <- 1e-12
LIGHT_LEAKAGE_EXPONENT <- 0.5
MODERATE_LEAKAGE_EXPONENT <- 1.0
SIMILARITY_METHOD <- "whole_grid_cosine_dual_leakage_penalty_v2"

PROFILES <- c("reference", "high_y_side", "low_y_side", "perimeter")
SCOPES <- c("full_game", "quarter_1", "quarter_2", "quarter_3", "quarter_4")

FROZEN_CONFIGURATIONS <- data.frame(
  performance_rank = 1:5,
  config_key = c(
    "b050_p02500_s250",
    "b050_p01000_s250",
    "b100_p02500_s250",
    "b100_p01000_s200",
    "b100_p01000_s250"
  ),
  n_bootstraps = c(50L, 50L, 100L, 100L, 100L),
  possessions_per_bootstrap = c(2500L, 1000L, 2500L, 1000L, 1000L),
  samples_per_possession = c(250L, 250L, 250L, 200L, 250L),
  stringsAsFactors = FALSE
)

abort <- function(..., call. = FALSE) {
  stop(sprintf(...), call. = call.)
}

collapse_messages <- function(x) {
  x <- unique(trimws(as.character(x)))
  x <- x[nzchar(x)]
  paste(gsub("[\r\n]+", " ", x), collapse = " | ")
}

parse_bool <- function(x, name) {
  value <- tolower(trimws(as.character(x)))
  if (value %in% c("1", "true", "t", "yes", "y")) return(TRUE)
  if (value %in% c("0", "false", "f", "no", "n")) return(FALSE)
  abort("%s must be true or false; got '%s'.", name, x)
}

split_filter <- function(x) {
  if (is.null(x) || !nzchar(trimws(x))) return(NULL)
  values <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  unique(values[nzchar(values)])
}

print_help <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript --vanilla scripts/run_profile_splines.R \\\n",
    "    --input-root PATH --output-root PATH [options]\n\n",
    "Required:\n",
    "  --input-root PATH     Directory containing config_manifest.csv,\n",
    "                        stream_manifest.csv, and streams/.\n",
    "  --output-root PATH    Study output directory.\n\n",
    "Options:\n",
    "  --method NAME         mgcv smoothing selection method [GCV.Cp].\n",
    "  --workers N           Requested PSOCK workers, capped at 4 and at\n",
    "                        the available physical cores [1].\n",
    "  --profiles CSV        Filter profile names.\n",
    "  --scopes CSV          Filter analysis scopes.\n",
    "  --configs CSV         Filter frozen config keys.\n",
    "  --stream-keys CSV     Filter exporter stream keys.\n",
    "  --resume BOOL         Reuse compatible atomic checkpoints [true].\n",
    "  --finalize BOOL       Publish canonical CSV/RDS outputs once all\n",
    "                        100 profile/scope/config rows exist [true].\n",
    "  --dry-run BOOL        Validate and print the selected work [false].\n",
    "  --help                Show this message.\n"
  ))
}

parse_cli <- function(args) {
  values <- list(
    input_root = NULL,
    output_root = NULL,
    method = DEFAULT_GAM_METHOD,
    workers = "1",
    profiles = NULL,
    scopes = NULL,
    configs = NULL,
    stream_keys = NULL,
    resume = "true",
    finalize = "true",
    dry_run = "false",
    help = "false"
  )
  aliases <- c(
    "input-root" = "input_root",
    "output-root" = "output_root",
    "stream-keys" = "stream_keys",
    "config-keys" = "configs",
    "dry-run" = "dry_run"
  )
  allowed <- names(values)
  i <- 1L
  while (i <= length(args)) {
    token <- args[[i]]
    if (!startsWith(token, "--")) abort("Unexpected argument: %s", token)
    token <- substring(token, 3L)
    if (grepl("=", token, fixed = TRUE)) {
      pieces <- strsplit(token, "=", fixed = TRUE)[[1]]
      key <- pieces[[1]]
      value <- paste(pieces[-1], collapse = "=")
    } else {
      key <- token
      if (key == "help") {
        value <- "true"
      } else {
        if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
          abort("Option --%s requires a value.", key)
        }
        i <- i + 1L
        value <- args[[i]]
      }
    }
    normalized <- if (key %in% names(aliases)) aliases[[key]] else gsub("-", "_", key)
    if (!normalized %in% allowed) abort("Unknown option: --%s", key)
    values[[normalized]] <- value
    i <- i + 1L
  }
  values$help <- parse_bool(values$help, "--help")
  if (values$help) return(values)
  if (is.null(values$input_root)) abort("--input-root is required.")
  if (is.null(values$output_root)) abort("--output-root is required.")
  values$workers <- suppressWarnings(as.integer(values$workers))
  if (is.na(values$workers) || values$workers < 1L) {
    abort("--workers must be a positive integer.")
  }
  values$resume <- parse_bool(values$resume, "--resume")
  values$finalize <- parse_bool(values$finalize, "--finalize")
  values$dry_run <- parse_bool(values$dry_run, "--dry-run")
  values$profiles <- split_filter(values$profiles)
  values$scopes <- split_filter(values$scopes)
  values$configs <- split_filter(values$configs)
  values$stream_keys <- split_filter(values$stream_keys)
  values$method <- trimws(values$method)
  supported_methods <- c("GCV.Cp", "GACV.Cp", "REML", "ML", "P-REML", "P-ML")
  if (!values$method %in% supported_methods) {
    abort(
      "Unsupported --method '%s'. Choose one of: %s.",
      values$method,
      paste(supported_methods, collapse = ", ")
    )
  }
  values
}

require_columns <- function(frame, columns, label) {
  missing <- setdiff(columns, names(frame))
  if (length(missing)) {
    abort("%s is missing required column(s): %s.", label, paste(missing, collapse = ", "))
  }
}

first_column <- function(frame, candidates, required = TRUE, label = "manifest") {
  found <- candidates[candidates %in% names(frame)]
  if (length(found)) return(found[[1]])
  if (required) {
    abort(
      "%s needs one of these columns: %s.",
      label,
      paste(candidates, collapse = ", ")
    )
  }
  NULL
}

column_values <- function(frame, candidates, required = TRUE, label = "manifest") {
  column <- first_column(frame, candidates, required = required, label = label)
  if (is.null(column)) return(rep(NA_character_, nrow(frame)))
  frame[[column]]
}

read_manifest <- function(path, label) {
  if (!file.exists(path)) abort("Missing %s: %s", label, path)
  frame <- tryCatch(
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) abort("Could not read %s '%s': %s", label, path, conditionMessage(e))
  )
  if (!nrow(frame)) abort("%s is empty: %s", label, path)
  frame
}

resolve_path <- function(root, path) {
  path <- as.character(path)
  if (length(path) != 1L || is.na(path) || !nzchar(path)) abort("Encountered an empty input path.")
  is_absolute <- grepl("^[A-Za-z]:[/\\\\]", path) || startsWith(path, "/") || startsWith(path, "\\\\")
  candidate <- if (is_absolute) path else file.path(root, path)
  normalizePath(candidate, winslash = "/", mustWork = TRUE)
}

normalize_token <- function(x) {
  gsub("[^a-z0-9]+", "", tolower(as.character(x)))
}

single_manifest_value <- function(frame, candidates, required = FALSE, label = "manifest") {
  column <- first_column(frame, candidates, required = required, label = label)
  if (is.null(column)) return(NA_character_)
  values <- unique(as.character(frame[[column]]))
  values <- values[!is.na(values) & nzchar(values)]
  if (length(values) != 1L) {
    abort("%s column %s must contain one consistent value.", label, column)
  }
  values[[1]]
}

canonicalize_config_manifest <- function(frame) {
  require_columns(
    frame,
    c(
      "performance_rank", "config_key", "n_bootstraps",
      "possessions_per_bootstrap", "samples_per_possession", "scope",
      "stream_key", "prefix_slots", "x_path",
      paste0("y_", PROFILES, "_path")
    ),
    "config_manifest.csv"
  )
  out <- data.frame(
    performance_rank = as.integer(column_values(frame, c("performance_rank", "rank"), FALSE)),
    config_key = as.character(column_values(frame, c("config_key"))),
    n_bootstraps = as.integer(column_values(frame, c("n_bootstraps", "bootstrap_count"))),
    possessions_per_bootstrap = as.integer(column_values(frame, c("possessions_per_bootstrap", "n_possessions"))),
    samples_per_possession = as.integer(column_values(frame, c("samples_per_possession", "n_samples_per_possession"))),
    scope = as.character(column_values(frame, c("scope"))),
    stream_key = as.character(column_values(frame, c("stream_key", "bootstrap_group_key"))),
    prefix_slots = as.integer(column_values(frame, c("prefix_slots"))),
    x_path = as.character(column_values(frame, c("x_path"))),
    stringsAsFactors = FALSE
  )
  for (profile in PROFILES) {
    out[[paste0("y_", profile, "_path")]] <- as.character(
      column_values(frame, c(paste0("y_", profile, "_path")))
    )
  }
  if (anyNA(out) || any(!nzchar(out$config_key)) || any(!nzchar(out$scope)) ||
      any(!nzchar(out$stream_key)) || any(!nzchar(out$x_path))) {
    abort("config_manifest.csv contains missing identity/settings values.")
  }
  if (any(out$n_bootstraps < 1L) || any(out$possessions_per_bootstrap < 1L) || any(out$samples_per_possession < 1L)) {
    abort("config_manifest.csv contains non-positive settings.")
  }
  out
}

canonicalize_stream_manifest <- function(frame) {
  required <- c(
    "stream_key", "simulation_tag", "simulated_games", "scope",
    "possessions_per_bootstrap", "samples_per_possession", "max_bootstraps",
    "n_cells", "config_keys", "seed_policy_version", "slot_manifest_path",
    "x_dtype", "x_layout", "column_order", "x_path", "x_shape",
    "x_nbytes", "x_sha256", "status", "completed_slots",
    "first_slot_equality_passed",
    unlist(lapply(PROFILES, function(profile) paste0(
      "y_", profile, c("_path", "_shape", "_nbytes", "_sha256")
    )), use.names = FALSE)
  )
  require_columns(frame, required, "stream_manifest.csv")
  hash_columns <- c("x_sha256", paste0("y_", PROFILES, "_sha256"))
  if (anyNA(frame[, setdiff(required, hash_columns), drop = FALSE])) {
    abort("stream_manifest.csv contains missing non-hash metadata.")
  }
  out <- data.frame(
    stream_key = as.character(column_values(frame, c("stream_key", "bootstrap_group_key"))),
    scope = as.character(column_values(frame, c("scope"))),
    possessions_per_bootstrap = as.integer(column_values(frame, c("possessions_per_bootstrap", "n_possessions"))),
    samples_per_possession = as.integer(column_values(frame, c("samples_per_possession", "n_samples_per_possession"))),
    max_bootstraps = as.integer(column_values(frame, c("max_bootstraps", "n_bootstraps"))),
    n_cells = as.integer(column_values(frame, c("n_cells", "n_features", "grid_cells"))),
    x_path = as.character(column_values(frame, c("x_path", "x_relative_path", "X_path"))),
    simulation_tag = as.character(frame$simulation_tag),
    simulated_games = as.integer(frame$simulated_games),
    config_keys = as.character(frame$config_keys),
    seed_policy_version = as.character(frame$seed_policy_version),
    slot_manifest_path = as.character(frame$slot_manifest_path),
    x_dtype = as.character(frame$x_dtype),
    x_layout = as.character(frame$x_layout),
    column_order = as.character(frame$column_order),
    x_shape = as.character(frame$x_shape),
    x_nbytes = as.double(frame$x_nbytes),
    x_sha256 = as.character(frame$x_sha256),
    status = as.character(frame$status),
    completed_slots = as.integer(frame$completed_slots),
    first_slot_equality_passed = as.character(frame$first_slot_equality_passed),
    stringsAsFactors = FALSE
  )
  # Retain any exporter-provided per-profile y path/size/checksum fields.
  for (name in setdiff(names(frame), names(out))) out[[name]] <- frame[[name]]
  if (anyNA(out[, c(
    "stream_key", "simulation_tag", "simulated_games", "scope",
    "possessions_per_bootstrap", "samples_per_possession", "max_bootstraps",
    "n_cells", "x_path", "x_nbytes", "completed_slots"
  )])) {
    abort("stream_manifest.csv contains missing required values.")
  }
  if (anyDuplicated(out$stream_key)) abort("stream_manifest.csv contains duplicate stream_key values.")
  out
}

validate_frozen_configurations <- function(config) {
  expected <- merge(
    FROZEN_CONFIGURATIONS[, -1L],
    data.frame(scope = SCOPES, stringsAsFactors = FALSE),
    by = NULL
  )
  key_columns <- c(
    "config_key", "n_bootstraps", "possessions_per_bootstrap",
    "samples_per_possession", "scope"
  )
  observed <- unique(config[, key_columns])
  expected_key <- do.call(paste, c(expected[key_columns], sep = "|"))
  observed_key <- do.call(paste, c(observed[key_columns], sep = "|"))
  if (length(observed_key) != length(expected_key) || !setequal(observed_key, expected_key)) {
    abort("config_manifest.csv does not contain exactly the frozen top-five configurations across all five scopes.")
  }
  if (anyDuplicated(config[, c("config_key", "scope")])) {
    abort("config_manifest.csv has duplicate config/scope rows.")
  }
  invisible(TRUE)
}

get_profile_field <- function(stream_row, profile, kind) {
  candidates <- switch(
    kind,
    path = c(
      paste0("y_", profile, "_path"),
      paste0("y__", profile, "_path"),
      paste0(profile, "_y_path")
    ),
    nbytes = c(
      paste0("y_", profile, "_nbytes"),
      paste0("y__", profile, "_nbytes"),
      paste0(profile, "_y_nbytes")
    ),
    shape = c(
      paste0("y_", profile, "_shape"),
      paste0("y__", profile, "_shape"),
      paste0(profile, "_y_shape")
    ),
    sha256 = c(
      paste0("y_", profile, "_sha256"),
      paste0("y__", profile, "_sha256"),
      paste0(profile, "_y_sha256")
    )
  )
  found <- candidates[candidates %in% names(stream_row)]
  if (!length(found)) return(NA_character_)
  as.character(stream_row[[found[[1]]]][[1]])
}

parse_manifest_bool <- function(value, label) {
  normalized <- normalize_token(value)
  if (length(normalized) != 1L || is.na(normalized)) {
    abort("%s must be one non-missing boolean value.", label)
  }
  if (normalized %in% c("true", "1")) return(TRUE)
  if (normalized %in% c("false", "0")) return(FALSE)
  abort("%s must be true or false; found '%s'.", label, as.character(value))
}

parse_shape <- function(value, dimensions, label) {
  pieces <- strsplit(tolower(trimws(as.character(value))), "x", fixed = TRUE)[[1]]
  parsed <- suppressWarnings(as.double(pieces))
  if (length(parsed) != dimensions || any(!is.finite(parsed)) ||
      any(parsed < 0) || any(parsed != floor(parsed))) {
    abort("%s has invalid shape '%s'.", label, as.character(value))
  }
  parsed
}

validate_sha256_text <- function(value, label) {
  value <- tolower(trimws(as.character(value)))
  if (length(value) != 1L || is.na(value) ||
      !grepl("^[0-9a-f]{64}$", value)) {
    abort("%s is not a valid SHA-256 digest.", label)
  }
  value
}

sha256_file <- function(path) {
  sha256sum <- get0("sha256sum", envir = asNamespace("tools"), inherits = FALSE)
  if (is.null(sha256sum)) {
    abort("This runner requires base R tools::sha256sum for input validation (R 4.5 or newer).")
  }
  result <- tryCatch(
    unname(sha256sum(path)),
    error = function(e) abort("Could not SHA-256 hash '%s': %s", path, conditionMessage(e))
  )
  validate_sha256_text(result, sprintf("Computed hash for %s", path))
}

sha256_bytes <- function(bytes) {
  sha256sum <- get0("sha256sum", envir = asNamespace("tools"), inherits = FALSE)
  if (is.null(sha256sum)) {
    abort("This runner requires base R tools::sha256sum for input validation (R 4.5 or newer).")
  }
  validate_sha256_text(unname(sha256sum(bytes = bytes)), "Computed byte hash")
}

expected_bootstrap_seed <- function(simulation_tag, scope, possessions, samples, slot_index) {
  payload <- paste(
    simulation_tag, SWEEP_DESIGN_VERSION, SEED_POLICY_VERSION, "bootstrap",
    scope, as.integer(possessions), as.integer(samples), as.integer(slot_index),
    sep = "|"
  )
  digest <- sha256_bytes(charToRaw(enc2utf8(payload)))
  first_bytes <- strtoi(
    substring(digest, c(1L, 3L, 5L, 7L), c(2L, 4L, 6L, 8L)),
    base = 16L
  )
  sum(as.double(first_bytes) * c(1, 256, 65536, 16777216))
}

normalized_relative_path <- function(path) {
  gsub("\\\\", "/", trimws(as.character(path)))
}

validate_slot_rows <- function(slot_manifest, stream_row) {
  required <- c(
    "stream_key", "scope", "possessions_per_bootstrap",
    "samples_per_possession", "slot_index", "bootstrap_seed",
    "x_byte_offset", "x_nbytes", "y_byte_offset", "y_nbytes",
    "seed_policy_version"
  )
  require_columns(slot_manifest, required, "slot_manifest.csv")
  key <- as.character(stream_row$stream_key[[1]])
  slots <- slot_manifest[as.character(slot_manifest$stream_key) == key, , drop = FALSE]
  max_b <- as.integer(stream_row$max_bootstraps[[1]])
  p <- as.integer(stream_row$possessions_per_bootstrap[[1]])
  if (nrow(slots) != max_b) {
    abort("slot_manifest.csv must contain exactly %d rows for %s; found %d.", max_b, key, nrow(slots))
  }
  numeric_columns <- c(
    "possessions_per_bootstrap", "samples_per_possession", "slot_index",
    "bootstrap_seed", "x_byte_offset", "x_nbytes", "y_byte_offset", "y_nbytes"
  )
  numbers <- lapply(numeric_columns, function(column) {
    value <- suppressWarnings(as.double(slots[[column]]))
    if (any(!is.finite(value)) || any(value < 0) || any(value != floor(value))) {
      abort("slot_manifest.csv column %s is invalid for %s.", column, key)
    }
    value
  })
  names(numbers) <- numeric_columns
  ordering <- order(numbers$slot_index, method = "radix")
  slots <- slots[ordering, , drop = FALSE]
  numbers <- lapply(numbers, function(value) value[ordering])
  expected_slot <- 0:(max_b - 1L)
  if (!identical(numbers$slot_index, as.double(expected_slot))) {
    abort("slot_manifest.csv does not contain every contiguous prefix slot for %s.", key)
  }
  expected_x_count <- as.double(p) * N_CELLS
  expected_y_count <- as.double(p)
  expected_seeds <- vapply(expected_slot, function(slot_index) {
    expected_bootstrap_seed(
      stream_row$simulation_tag[[1]], stream_row$scope[[1]], p,
      stream_row$samples_per_possession[[1]], slot_index
    )
  }, numeric(1))
  checks <- list(
    scope = as.character(slots$scope) == as.character(stream_row$scope[[1]]),
    possessions = numbers$possessions_per_bootstrap == p,
    samples = numbers$samples_per_possession == as.integer(stream_row$samples_per_possession[[1]]),
    seed_policy = as.character(slots$seed_policy_version) == SEED_POLICY_VERSION,
    x_offset = numbers$x_byte_offset == as.double(expected_slot) * expected_x_count,
    x_nbytes = numbers$x_nbytes == expected_x_count,
    y_offset = numbers$y_byte_offset == as.double(expected_slot) * expected_y_count,
    y_nbytes = numbers$y_nbytes == expected_y_count,
    bootstrap_seed = numbers$bootstrap_seed == expected_seeds
  )
  failed <- names(checks)[!vapply(checks, all, logical(1))]
  if (length(failed)) {
    abort("slot_manifest.csv metadata mismatch for %s: %s.", key, paste(failed, collapse = ", "))
  }
  invisible(TRUE)
}

validate_manifests <- function(input_root, raw_config, raw_stream) {
  config <- canonicalize_config_manifest(raw_config)
  stream <- canonicalize_stream_manifest(raw_stream)
  validate_frozen_configurations(config)

  if (!setequal(unique(config$scope), SCOPES)) abort("Unexpected analysis scopes in config_manifest.csv.")
  expected_ranks <- FROZEN_CONFIGURATIONS$performance_rank[
    match(config$config_key, FROZEN_CONFIGURATIONS$config_key)
  ]
  if (any(config$performance_rank != expected_ranks)) {
    abort("config_manifest.csv performance_rank values do not match the frozen QUT ranking.")
  }
  if (any(config$prefix_slots != config$n_bootstraps)) {
    abort("config_manifest.csv prefix_slots must equal n_bootstraps.")
  }
  expected_stream_key <- sprintf(
    "%s__p%05d_s%03d", config$scope,
    config$possessions_per_bootstrap, config$samples_per_possession
  )
  if (any(config$stream_key != expected_stream_key)) {
    abort("config_manifest.csv contains a stream_key inconsistent with scope/P/S.")
  }
  if (!all(config$stream_key %in% stream$stream_key)) abort("A config row references an unknown stream_key.")
  expected_stream_keys <- unique(config$stream_key)
  if (nrow(stream) != length(expected_stream_keys) ||
      !setequal(stream$stream_key, expected_stream_keys)) {
    abort("stream_manifest.csv must contain exactly the 15 frozen scope/P/S streams.")
  }
  joined <- merge(
    config,
    stream[, c("stream_key", "scope", "possessions_per_bootstrap", "samples_per_possession", "max_bootstraps", "n_cells")],
    by = "stream_key",
    suffixes = c("_config", "_stream"),
    all.x = TRUE,
    sort = FALSE
  )
  if (any(joined$scope_config != joined$scope_stream) ||
      any(joined$possessions_per_bootstrap_config != joined$possessions_per_bootstrap_stream) ||
      any(joined$samples_per_possession_config != joined$samples_per_possession_stream)) {
    abort("Config and stream manifests disagree on scope/P/S metadata.")
  }
  if (any(joined$n_cells != N_CELLS)) abort("Every stream must contain exactly %d cells.", N_CELLS)
  if (any(joined$max_bootstraps < joined$n_bootstraps)) abort("A stream does not contain all required prefix slots.")

  simulation_tag <- single_manifest_value(stream, c("simulation_tag"), FALSE, "stream_manifest.csv")
  if (is.na(simulation_tag)) {
    simulation_tag <- single_manifest_value(raw_config, c("simulation_tag"), FALSE, "config_manifest.csv")
  }
  simulated_games <- suppressWarnings(as.integer(single_manifest_value(stream, c("simulated_games", "n_simulated_games"), FALSE)))
  if (is.na(simulated_games)) {
    simulated_games <- suppressWarnings(as.integer(single_manifest_value(raw_config, c("simulated_games", "n_simulated_games"), FALSE)))
  }
  if (is.na(simulated_games) && !is.na(simulation_tag)) {
    hit <- regmatches(simulation_tag, regexec("^games([0-9]+)", simulation_tag))[[1]]
    if (length(hit) == 2L) simulated_games <- as.integer(hit[[2]])
  }
  if (is.na(simulated_games) || !simulated_games %in% c(8L, 10L)) {
    abort("Could not validate simulated_games as 8 or 10 from the manifests.")
  }
  if (is.na(simulation_tag) || !nzchar(simulation_tag)) abort("The manifests must provide simulation_tag.")

  allowed_status <- c("pending", "partial", "valid")
  statuses <- normalize_token(stream$status)
  if (any(!statuses %in% allowed_status)) {
    abort("stream_manifest.csv status must be pending, partial, or valid.")
  }
  if (any(stream$simulated_games != simulated_games) ||
      any(stream$simulation_tag != simulation_tag)) {
    abort("stream_manifest.csv contains inconsistent study identity metadata.")
  }
  if (any(stream$seed_policy_version != SEED_POLICY_VERSION)) {
    abort("stream_manifest.csv has an unexpected seed_policy_version.")
  }
  if (any(normalize_token(stream$x_dtype) != "uint8")) {
    abort("X dtype must be uint8 for every stream.")
  }
  if (any(normalize_token(stream$x_layout) != "cslotpossessionflatindex")) {
    abort("X layout must be 'C: slot, possession, flat_index'.")
  }
  if (any(normalize_token(stream$column_order) != "xmajoryfastflatindexxbin10ybin")) {
    abort("Feature order must be x-major/y-fast with flat_index=x_bin*10+y_bin.")
  }
  if (any(stream$n_cells != N_CELLS) || any(stream$max_bootstraps != 100L)) {
    abort("Every frozen stream must declare shape B=100 with %d cells.", N_CELLS)
  }
  for (i in seq_len(nrow(stream))) {
    row <- stream[i, , drop = FALSE]
    key <- row$stream_key[[1]]
    p <- as.integer(row$possessions_per_bootstrap[[1]])
    max_b <- as.integer(row$max_bootstraps[[1]])
    expected_key <- sprintf("%s__p%05d_s%03d", row$scope[[1]], p, row$samples_per_possession[[1]])
    if (key != expected_key) abort("Malformed stream_key in stream_manifest.csv: %s.", key)
    expected_configs <- config$config_key[config$stream_key == key]
    observed_configs <- trimws(strsplit(row$config_keys[[1]], ";", fixed = TRUE)[[1]])
    if (!setequal(observed_configs, expected_configs) || length(observed_configs) != length(expected_configs)) {
      abort("stream_manifest.csv config_keys mismatch for %s.", key)
    }
    expected_x <- as.double(max_b) * p * N_CELLS
    if (!identical(parse_shape(row$x_shape[[1]], 3L, paste0(key, " x_shape")), c(as.double(max_b), as.double(p), as.double(N_CELLS))) ||
        row$x_nbytes[[1]] != expected_x) {
      abort("stream_manifest.csv X shape/size metadata mismatch for %s.", key)
    }
    completed <- as.integer(row$completed_slots[[1]])
    equality_passed <- parse_manifest_bool(row$first_slot_equality_passed[[1]], paste0(key, " first_slot_equality_passed"))
    if (completed < 0L || completed > max_b ||
        (statuses[[i]] == "valid" && (completed != max_b || !equality_passed)) ||
        (statuses[[i]] == "pending" && (completed != 0L || equality_passed)) ||
        (statuses[[i]] == "partial" && completed > 0L && !equality_passed)) {
      abort("stream_manifest.csv completion metadata is inconsistent for %s.", key)
    }
    config_rows <- config[config$stream_key == key, , drop = FALSE]
    if (any(normalized_relative_path(config_rows$x_path) != normalized_relative_path(row$x_path[[1]]))) {
      abort("config_manifest.csv and stream_manifest.csv X paths disagree for %s.", key)
    }
    for (profile in PROFILES) {
      expected_y <- as.double(max_b) * p
      y_shape <- parse_shape(get_profile_field(row, profile, "shape"), 2L, paste0(key, "/", profile, " y_shape"))
      y_nbytes <- suppressWarnings(as.double(get_profile_field(row, profile, "nbytes")))
      if (!identical(y_shape, c(as.double(max_b), as.double(p))) ||
          !is.finite(y_nbytes) || y_nbytes != expected_y) {
        abort("stream_manifest.csv y shape/size metadata mismatch for %s/%s.", key, profile)
      }
      config_y <- config_rows[[paste0("y_", profile, "_path")]]
      stream_y <- get_profile_field(row, profile, "path")
      if (any(normalized_relative_path(config_y) != normalized_relative_path(stream_y))) {
        abort("Config and stream y paths disagree for %s/%s.", key, profile)
      }
    }
  }

  global_metadata <- validate_global_metadata(input_root, simulation_tag, simulated_games)
  planted <- load_planted_profiles(input_root)

  list(
    config = config,
    stream = stream,
    simulation_tag = simulation_tag,
    simulated_games = simulated_games,
    global_metadata = global_metadata,
    planted_profiles = planted
  )
}

validate_selected_streams <- function(input_root, manifests, stream_keys) {
  stream <- manifests$stream[manifests$stream$stream_key %in% stream_keys, , drop = FALSE]
  if (nrow(stream) != length(unique(stream_keys))) abort("Could not locate every selected stream.")
  slot_cache <- new.env(parent = emptyenv())
  for (i in seq_len(nrow(stream))) {
    key <- stream$stream_key[[i]]
    if (normalize_token(stream$status[[i]]) != "valid") {
      abort("Selected stream %s is not valid (status=%s).", key, stream$status[[i]])
    }
    row <- stream[i, , drop = FALSE]
    slot_path <- resolve_path(input_root, row$slot_manifest_path[[1]])
    if (!exists(slot_path, envir = slot_cache, inherits = FALSE)) {
      assign(slot_path, read_manifest(slot_path, "slot manifest"), envir = slot_cache)
    }
    validate_slot_rows(get(slot_path, envir = slot_cache, inherits = FALSE), row)
    stream$resolved_slot_manifest_path[[i]] <- slot_path
    stream$slot_manifest_sha256[[i]] <- sha256_file(slot_path)

    expected_x <- as.double(row$max_bootstraps[[1]]) *
      as.double(row$possessions_per_bootstrap[[1]]) * N_CELLS
    x_path <- resolve_path(input_root, row$x_path[[1]])
    actual_x <- as.double(file.info(x_path)$size)
    if (!is.finite(actual_x) || actual_x != expected_x || row$x_nbytes[[1]] != expected_x) {
      abort("X file size mismatch for %s: expected %.0f bytes, found %.0f.", key, expected_x, actual_x)
    }
    expected_x_hash <- validate_sha256_text(row$x_sha256[[1]], paste0(key, " x_sha256"))
    actual_x_hash <- sha256_file(x_path)
    if (!identical(actual_x_hash, expected_x_hash)) abort("X SHA-256 mismatch for %s.", key)
    stream$x_path[[i]] <- x_path
    stream$x_sha256[[i]] <- actual_x_hash

    for (profile in PROFILES) {
      y_path <- resolve_path(input_root, get_profile_field(row, profile, "path"))
      expected_y <- as.double(row$max_bootstraps[[1]]) * as.double(row$possessions_per_bootstrap[[1]])
      actual_y <- as.double(file.info(y_path)$size)
      if (!is.finite(actual_y) || actual_y != expected_y) {
        abort("y file size mismatch for %s/%s: expected %.0f bytes, found %.0f.", key, profile, expected_y, actual_y)
      }
      expected_y_hash <- validate_sha256_text(
        get_profile_field(row, profile, "sha256"), paste0(key, "/", profile, " y_sha256")
      )
      actual_y_hash <- sha256_file(y_path)
      if (!identical(actual_y_hash, expected_y_hash)) abort("y SHA-256 mismatch for %s/%s.", key, profile)
      stream[[paste0("resolved_y_", profile)]][[i]] <- y_path
      stream[[paste0("resolved_y_", profile, "_sha256")]][[i]] <- actual_y_hash
    }
  }
  rownames(stream) <- NULL
  stream
}

build_basis <- function() {
  pixels <- data.frame(
    x = rep(0:(NX - 1L), each = NY),
    y = rep(0:(NY - 1L), times = NX)
  )
  stopifnot(
    identical(pixels$x, rep(0:(NX - 1L), each = NY)),
    identical(pixels$y, rep(0:(NY - 1L), times = NX))
  )
  smooth <- mgcv::smoothCon(
    mgcv::s(x, y, bs = "tp", k = BASIS_K),
    data = pixels,
    absorb.cons = FALSE
  )[[1]]
  basis <- smooth$X
  penalty <- smooth$S[[1]]
  if (!identical(dim(basis), c(N_CELLS, BASIS_K))) abort("Unexpected thin-plate basis dimensions.")
  if (!identical(dim(penalty), c(BASIS_K, BASIS_K))) abort("Unexpected thin-plate penalty dimensions.")
  if (length(smooth$rank) != 1L || smooth$rank < 1L) abort("Unexpected thin-plate penalty rank.")
  list(
    B = basis,
    S = penalty,
    rank = as.integer(smooth$rank),
    null_space_dimension = as.integer(smooth$null.space.dim),
    pixels = pixels,
    id = BASIS_ID
  )
}

planted_profiles <- function() {
  blank <- function() matrix(0, nrow = NX, ncol = NY)

  reference <- blank()
  reference[11:18, 1:3] <- 0.05
  reference[19:20, 1:3] <- 0.05
  reference[19:20, 7] <- 0.05
  reference[11:20, 7:10] <- 0.05
  reference[19:20, 3:6] <- 0.05
  reference[11:18, 3:6] <- 0
  reference[11:12, 7:8] <- 0
  reference[11:13, ] <- 0
  reference[19:20, 3:4] <- 0.05

  high_y_side <- blank()
  high_y_side[14:20, 6:10] <- 0.05

  low_y_side <- blank()
  low_y_side[14:20, 1:5] <- 0.05

  perimeter <- blank()
  perimeter[16:20, 10] <- 0.05
  perimeter[15:18, 9:10] <- 0.05
  perimeter[15:16, 8:9] <- 0.05
  perimeter[14:15, 4:7] <- 0.05
  perimeter[15:20, 1] <- 0.05
  perimeter[15:18, 2] <- 0.05
  perimeter[15:16, 3] <- 0.05
  perimeter[14, 1:10] <- 0.05
  perimeter[16:20, 4:7] <- -0.01
  perimeter[19:20, 4:7] <- -0.01
  perimeter[17:18, 4:7] <- -0.01
  perimeter[16, 4:7] <- 0

  maps <- list(
    reference = reference,
    high_y_side = high_y_side,
    low_y_side = low_y_side,
    perimeter = perimeter
  )
  lapply(maps, function(map) as.vector(t(map)))
}

validate_global_metadata <- function(input_root, simulation_tag, simulated_games) {
  path <- file.path(input_root, "metadata.json")
  if (!file.exists(path)) abort("Missing global metadata: %s", path)
  text <- paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  required_fragments <- c(
    sprintf('"format_version": "%s"', INPUT_FORMAT_VERSION),
    sprintf('"simulation_tag": "%s"', simulation_tag),
    sprintf('"n_simulated_games": %d', simulated_games),
    sprintf('"sweep_design_version": "%s"', SWEEP_DESIGN_VERSION),
    sprintf('"seed_policy_version": "%s"', SEED_POLICY_VERSION),
    '"comparison_label": "top5_tp_k20"',
    '"map_estimator": "nested_prefix_median_v1"',
    '"n_unique_streams": 15',
    '"n_cells": 200',
    '"x_bins": 20',
    '"y_bins": 10',
    '"flat_index": "x_bin * 10 + y_bin"',
    '"basis": "thin_plate"',
    '"k": 20'
  )
  missing <- required_fragments[!vapply(
    required_fragments, function(fragment) grepl(fragment, text, fixed = TRUE), logical(1)
  )]
  if (length(missing)) {
    abort("metadata.json does not match the required spline input contract: %s.", paste(missing, collapse = ", "))
  }
  list(path = normalizePath(path, winslash = "/", mustWork = TRUE), sha256 = sha256_file(path))
}

load_planted_profiles <- function(input_root) {
  path <- file.path(input_root, "planted_profiles.csv")
  frame <- read_manifest(path, "planted profile table")
  require_columns(
    frame,
    c(
      "profile", "profile_label", "profile_seed", "x_bin", "y_bin",
      "flat_index", "movement_effect", "end_effect"
    ),
    "planted_profiles.csv"
  )
  if (nrow(frame) != length(PROFILES) * N_CELLS ||
      !setequal(unique(as.character(frame$profile)), PROFILES)) {
    abort("planted_profiles.csv must contain exactly %d cells for each frozen profile.", N_CELLS)
  }
  expected_maps <- planted_profiles()
  maps <- list()
  for (profile in PROFILES) {
    rows <- frame[as.character(frame$profile) == profile, , drop = FALSE]
    numeric_columns <- c("profile_seed", "x_bin", "y_bin", "flat_index", "movement_effect", "end_effect")
    numeric_values <- lapply(numeric_columns, function(column) suppressWarnings(as.double(rows[[column]])))
    names(numeric_values) <- numeric_columns
    if (nrow(rows) != N_CELLS || any(!vapply(numeric_values, function(value) all(is.finite(value)), logical(1)))) {
      abort("planted_profiles.csv contains invalid rows for %s.", profile)
    }
    ordering <- order(numeric_values$flat_index, method = "radix")
    numeric_values <- lapply(numeric_values, function(value) value[ordering])
    expected_index <- 0:(N_CELLS - 1L)
    if (!identical(numeric_values$flat_index, as.double(expected_index)) ||
        any(numeric_values$x_bin != floor(expected_index / NY)) ||
        any(numeric_values$y_bin != expected_index %% NY)) {
      abort("planted_profiles.csv cell ordering is not x-major/y-fast for %s.", profile)
    }
    movement <- numeric_values$movement_effect
    if (any(abs(movement - expected_maps[[profile]]) > 1e-15) ||
        any(abs(numeric_values$end_effect) > 1e-15)) {
      abort("planted_profiles.csv does not match the frozen profile definition for %s.", profile)
    }
    maps[[profile]] <- movement
  }
  list(
    maps = maps,
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    sha256 = sha256_file(path)
  )
}

clip_value <- function(x, low, high) min(max(x, low), high)

profile_map_similarity <- function(planted, recovered) {
  planted <- as.double(planted)
  recovered <- as.double(recovered)
  if (length(planted) != N_CELLS || length(recovered) != N_CELLS) abort("Profile maps must have %d cells.", N_CELLS)
  if (any(!is.finite(planted)) || any(!is.finite(recovered))) abort("Profile maps contain non-finite values.")
  support <- abs(planted) > SUPPORT_TOLERANCE
  if (!any(support)) abort("Planted profile has no supported cells.")
  outside <- !support
  background <- if (any(outside)) median(recovered[outside]) else 0
  adjusted <- recovered - background
  planted_support <- planted[support]
  recovered_support <- adjusted[support]
  recovered_outside <- adjusted[outside]
  support_dot <- sum(planted_support * recovered_support)
  whole_dot <- sum(planted * adjusted)
  support_planted_norm <- sqrt(sum(planted_support^2))
  whole_planted_norm <- sqrt(sum(planted^2))
  support_energy <- sum(recovered_support^2)
  outside_energy <- sum(recovered_outside^2)
  total_energy <- support_energy + outside_energy
  support_cosine <- if (support_planted_norm > 0 && support_energy > 0) {
    support_dot / (support_planted_norm * sqrt(support_energy))
  } else 0
  outside_fraction <- if (total_energy > 0) outside_energy / total_energy else 0
  whole_cosine <- if (whole_planted_norm > 0 && total_energy > 0) {
    whole_dot / (whole_planted_norm * sqrt(total_energy))
  } else 0
  whole_cosine <- clip_value(whole_cosine, -1, 1)
  nonnegative_cosine <- max(0, whole_cosine)
  retained <- 1 - outside_fraction
  data.frame(
    similarity_method = SIMILARITY_METHOD,
    profile_support_sectors = sum(support),
    off_profile_sectors = sum(outside),
    recovered_background_level = background,
    light_leakage_exponent = LIGHT_LEAKAGE_EXPONENT,
    moderate_leakage_exponent = MODERATE_LEAKAGE_EXPONENT,
    whole_grid_cosine = whole_cosine,
    support_cosine = clip_value(support_cosine, -1, 1),
    outside_energy_fraction = clip_value(outside_fraction, 0, 1),
    lightly_penalized_whole_grid_cosine = clip_value(
      nonnegative_cosine * retained^LIGHT_LEAKAGE_EXPONENT, 0, 1
    ),
    moderately_penalized_whole_grid_cosine = clip_value(
      nonnegative_cosine * retained^MODERATE_LEAKAGE_EXPONENT, 0, 1
    ),
    recovered_support_energy = support_energy,
    recovered_outside_energy = outside_energy,
    recovered_dynamic_range = as.double(
      quantile(adjusted, 0.95, names = FALSE, type = 7) -
        quantile(adjusted, 0.05, names = FALSE, type = 7)
    ),
    stringsAsFactors = FALSE
  )
}

read_u8_slot <- function(connection, offset, count, label) {
  seek(connection, where = as.double(offset), origin = "start", rw = "read")
  bytes <- readBin(connection, what = "raw", n = as.integer(count))
  if (length(bytes) != count) {
    abort("Short read for %s: expected %d bytes, received %d.", label, count, length(bytes))
  }
  as.integer(bytes)
}

format_numbers <- function(x) {
  if (!length(x)) return("")
  paste(formatC(as.double(x), digits = 17L, format = "g"), collapse = ";")
}

fit_spline_slot <- function(X, y, basis, method) {
  warnings_seen <- character()
  started <- proc.time()[["elapsed"]]
  payload <- tryCatch(
    withCallingHandlers({
      Z <- X %*% basis$B
      design_rank <- qr(Z)$rank
      fit <- mgcv::gam(
        y ~ -1 + Z,
        data = list(y = y, Z = Z),
        family = binomial(link = "logit"),
        paraPen = list(Z = list(basis$S, rank = basis$rank)),
        method = method
      )
      beta <- drop(basis$B %*% coef(fit))
      if (length(beta) != N_CELLS || any(!is.finite(beta))) abort("mgcv returned a non-finite or incorrectly sized map.")
      smoothing <- as.double(fit$sp)
      if (length(smoothing) != 1L || !is.finite(smoothing) || smoothing < 0) abort("mgcv returned an invalid smoothing parameter.")
      converged <- isTRUE(fit$converged)
      if (!converged) abort("mgcv IRLS did not converge.")
      outer_convergence <- if (!is.null(fit$outer.info$conv)) as.character(fit$outer.info$conv) else ""
      list(
        ok = TRUE,
        beta = beta,
        diagnostic = list(
          status = "ok",
          warning = collapse_messages(warnings_seen),
          error = "",
          smoothing_parameters = format_numbers(smoothing),
          effective_degrees_freedom = sum(as.double(fit$edf)),
          design_rank = as.integer(design_rank),
          model_rank = as.integer(fit$rank),
          converged = converged,
          outer_convergence = collapse_messages(outer_convergence),
          boundary = isTRUE(fit$boundary),
          iterations = as.integer(fit$iter),
          deviance = as.double(fit$deviance),
          null_deviance = as.double(fit$null.deviance),
          deviance_explained = if (is.finite(fit$null.deviance) && fit$null.deviance != 0) {
            1 - fit$deviance / fit$null.deviance
          } else NA_real_,
          aic = as.double(AIC(fit)),
          log_likelihood = as.double(logLik(fit)),
          gcv_ubre = as.double(fit$gcv.ubre),
          map_min = min(beta),
          map_max = max(beta)
        )
      )
    }, warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }),
    error = function(e) list(
      ok = FALSE,
      beta = NULL,
      diagnostic = list(
        status = "failed",
        warning = collapse_messages(warnings_seen),
        error = conditionMessage(e)
      )
    )
  )
  payload$diagnostic$elapsed_seconds <- proc.time()[["elapsed"]] - started
  payload
}

metadata_signature <- function(metadata) {
  paste(
    names(metadata),
    vapply(metadata, function(x) paste(as.character(x), collapse = ";"), character(1)),
    sep = "=",
    collapse = "\n"
  )
}

checkpoint_backup_path <- function(path) paste0(path, ".bak")

recover_checkpoint_path <- function(path) {
  backup <- checkpoint_backup_path(path)
  if (!file.exists(path) && file.exists(backup)) {
    if (!file.rename(backup, path)) abort("Could not restore checkpoint backup: %s", backup)
  }
  invisible(path)
}

atomic_save_rds <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(pattern = paste0(".", basename(path), "-"), tmpdir = dirname(path), fileext = ".tmp")
  on.exit(if (file.exists(temporary)) unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(object, temporary, compress = FALSE)
  invisible(readRDS(temporary))
  backup <- checkpoint_backup_path(path)
  if (file.exists(backup)) unlink(backup, force = TRUE)
  if (file.exists(path) && !file.rename(path, backup)) abort("Could not rotate checkpoint: %s", path)
  if (!file.rename(temporary, path)) {
    if (!file.exists(path) && file.exists(backup)) file.rename(backup, path)
    abort("Could not publish checkpoint: %s", path)
  }
  if (file.exists(backup)) unlink(backup, force = TRUE)
  invisible(path)
}

atomic_write_csv <- function(frame, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(pattern = paste0(".", basename(path), "-"), tmpdir = dirname(path), fileext = ".tmp")
  on.exit(if (file.exists(temporary)) unlink(temporary, force = TRUE), add = TRUE)
  write.csv(frame, temporary, row.names = FALSE, na = "")
  backup <- checkpoint_backup_path(path)
  if (file.exists(backup)) unlink(backup, force = TRUE)
  if (file.exists(path) && !file.rename(path, backup)) abort("Could not rotate CSV: %s", path)
  if (!file.rename(temporary, path)) {
    if (!file.exists(path) && file.exists(backup)) file.rename(backup, path)
    abort("Could not publish CSV: %s", path)
  }
  if (file.exists(backup)) unlink(backup, force = TRUE)
  invisible(path)
}

new_checkpoint_state <- function(metadata, max_bootstraps) {
  list(
    version = SCRIPT_VERSION,
    metadata = metadata,
    signature = metadata_signature(metadata),
    maps = matrix(NA_real_, nrow = N_CELLS, ncol = max_bootstraps),
    done = rep(FALSE, max_bootstraps),
    attempts = integer(max_bootstraps),
    diagnostics = vector("list", max_bootstraps),
    updated_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )
}

load_checkpoint_state <- function(path, metadata, max_bootstraps, resume) {
  recover_checkpoint_path(path)
  if (!resume || !file.exists(path)) return(new_checkpoint_state(metadata, max_bootstraps))
  state <- tryCatch(readRDS(path), error = function(e) abort("Unreadable checkpoint %s: %s", path, conditionMessage(e)))
  expected_signature <- metadata_signature(metadata)
  if (!identical(state$version, SCRIPT_VERSION) ||
      !identical(state$signature, expected_signature)) {
    abort("Checkpoint metadata mismatch: %s", path)
  }
  if (!identical(dim(state$maps), c(N_CELLS, max_bootstraps)) ||
      length(state$done) != max_bootstraps || length(state$attempts) != max_bootstraps ||
      length(state$diagnostics) != max_bootstraps) {
    abort("Checkpoint dimensions are invalid: %s", path)
  }
  for (slot in which(state$done)) {
    if (any(!is.finite(state$maps[, slot]))) {
      state$done[[slot]] <- FALSE
      state$diagnostics[[slot]] <- NULL
    }
  }
  state
}

checkpoint_path_for_task <- function(output_root, method, profile, stream_key) {
  safe_method <- gsub("[^A-Za-z0-9_.-]", "_", method)
  safe_stream <- gsub("[^A-Za-z0-9_.-]", "_", stream_key)
  file.path(output_root, "checkpoints", paste0("method-", safe_method), profile, paste0(safe_stream, ".rds"))
}

build_task <- function(stream_row, profile, required_bootstraps, context) {
  p <- as.integer(stream_row$possessions_per_bootstrap[[1]])
  max_b <- as.integer(stream_row$max_bootstraps[[1]])
  x_path <- as.character(stream_row$x_path[[1]])
  y_path <- as.character(stream_row[[paste0("resolved_y_", profile)]][[1]])
  x_size <- as.double(file.info(x_path)$size)
  y_size <- as.double(file.info(y_path)$size)
  metadata <- list(
    script_version = SCRIPT_VERSION,
    input_format_version = INPUT_FORMAT_VERSION,
    input_metadata_sha256 = context$input_metadata_sha256,
    planted_profiles_sha256 = context$planted_profiles_sha256,
    simulation_tag = context$simulation_tag,
    simulated_games = as.integer(context$simulated_games),
    profile = profile,
    scope = as.character(stream_row$scope[[1]]),
    stream_key = as.character(stream_row$stream_key[[1]]),
    possessions_per_bootstrap = p,
    samples_per_possession = as.integer(stream_row$samples_per_possession[[1]]),
    max_bootstraps = max_b,
    n_cells = N_CELLS,
    x_path = x_path,
    x_nbytes = x_size,
    x_sha256 = as.character(stream_row$x_sha256[[1]]),
    y_path = y_path,
    y_nbytes = y_size,
    y_sha256 = as.character(stream_row[[paste0("resolved_y_", profile, "_sha256")]][[1]]),
    slot_manifest_path = as.character(stream_row$resolved_slot_manifest_path[[1]]),
    slot_manifest_sha256 = as.character(stream_row$slot_manifest_sha256[[1]]),
    raw_dtype = "uint8",
    raw_layout = "slot-major,row-major",
    feature_order = "x-major,y-fast,index=x*10+y",
    basis_id = BASIS_ID,
    basis_k = BASIS_K,
    absorb_constraints = FALSE,
    explicit_intercept = FALSE,
    family = "binomial(logit)",
    gam_method = context$method,
    r_version = as.character(getRversion()),
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )
  list(
    profile = profile,
    scope = metadata$scope,
    stream_key = metadata$stream_key,
    possessions = p,
    samples = metadata$samples_per_possession,
    max_bootstraps = max_b,
    required_bootstraps = as.integer(required_bootstraps),
    x_path = x_path,
    y_path = y_path,
    checkpoint_path = checkpoint_path_for_task(context$output_root, context$method, profile, metadata$stream_key),
    metadata = metadata,
    resume = context$resume,
    method = context$method
  )
}

process_stream_task <- function(task, basis) {
  state <- load_checkpoint_state(
    task$checkpoint_path, task$metadata, task$max_bootstraps, task$resume
  )
  required_slots <- seq_len(task$required_bootstraps)
  pending <- required_slots[!state$done[required_slots]]
  if (!length(pending)) {
    return(list(task = task$stream_key, profile = task$profile, complete = TRUE, failed_slots = integer()))
  }
  x_connection <- file(task$x_path, open = "rb")
  y_connection <- file(task$y_path, open = "rb")
  on.exit(close(x_connection), add = TRUE)
  on.exit(close(y_connection), add = TRUE)

  for (slot in pending) {
    state$attempts[[slot]] <- state$attempts[[slot]] + 1L
    slot_started <- proc.time()[["elapsed"]]
    outcome <- tryCatch({
      x_offset <- as.double(slot - 1L) * task$possessions * N_CELLS
      y_offset <- as.double(slot - 1L) * task$possessions
      x_values <- read_u8_slot(
        x_connection, x_offset, task$possessions * N_CELLS,
        sprintf("%s/%s X slot %d", task$stream_key, task$profile, slot - 1L)
      )
      y <- read_u8_slot(
        y_connection, y_offset, task$possessions,
        sprintf("%s/%s y slot %d", task$stream_key, task$profile, slot - 1L)
      )
      if (any(!x_values %in% 0:1)) abort("X slot contains values outside {0,1}.")
      if (any(!y %in% 0:1)) abort("y slot contains values outside {0,1}.")
      if (length(unique(y)) != 2L) abort("y slot does not contain both outcome classes.")
      X <- matrix(x_values, nrow = task$possessions, ncol = N_CELLS, byrow = TRUE)
      if (any(rowSums(X) < 1L)) abort("X slot contains an all-zero possession row.")
      fit <- fit_spline_slot(X, y, basis, task$method)
      fit$diagnostic$n_rows <- task$possessions
      fit$diagnostic$n_positive <- sum(y)
      fit$diagnostic$outcome_rate <- mean(y)
      fit$diagnostic$slot_elapsed_seconds <- proc.time()[["elapsed"]] - slot_started
      fit
    }, error = function(e) list(
      ok = FALSE,
      beta = NULL,
      diagnostic = list(
        status = "failed",
        warning = "",
        error = conditionMessage(e),
        elapsed_seconds = NA_real_,
        slot_elapsed_seconds = proc.time()[["elapsed"]] - slot_started
      )
    ))
    diagnostic <- c(
      list(
        slot_index = slot - 1L,
        attempts = state$attempts[[slot]],
        profile = task$profile,
        scope = task$scope,
        stream_key = task$stream_key,
        possessions_per_bootstrap = task$possessions,
        samples_per_possession = task$samples,
        x_offset_bytes = as.double(slot - 1L) * task$possessions * N_CELLS,
        y_offset_bytes = as.double(slot - 1L) * task$possessions
      ),
      outcome$diagnostic
    )
    state$diagnostics[[slot]] <- diagnostic
    if (isTRUE(outcome$ok)) {
      state$maps[, slot] <- outcome$beta
      state$done[[slot]] <- TRUE
    } else {
      state$maps[, slot] <- NA_real_
      state$done[[slot]] <- FALSE
    }
    state$updated_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    atomic_save_rds(state, task$checkpoint_path)
    rm(outcome)
    gc(verbose = FALSE)
  }
  failed <- required_slots[!state$done[required_slots]] - 1L
  list(
    task = task$stream_key,
    profile = task$profile,
    complete = !length(failed),
    failed_slots = failed
  )
}

list_rows_to_frame <- function(rows) {
  if (!length(rows)) return(data.frame())
  columns <- unique(unlist(lapply(rows, names), use.names = FALSE))
  normalized <- lapply(rows, function(row) {
    values <- setNames(vector("list", length(columns)), columns)
    for (column in columns) values[[column]] <- if (column %in% names(row)) row[[column]] else NA
    as.data.frame(values, stringsAsFactors = FALSE)
  })
  do.call(rbind, normalized)
}

make_all_tasks <- function(context, manifests, profiles = PROFILES, config = manifests$config) {
  requested <- aggregate(
    n_bootstraps ~ stream_key,
    data = config,
    FUN = max
  )
  tasks <- list()
  for (i in seq_len(nrow(requested))) {
    row <- manifests$stream[manifests$stream$stream_key == requested$stream_key[[i]], , drop = FALSE]
    if (nrow(row) != 1L) abort("Expected one stream row for %s.", requested$stream_key[[i]])
    for (profile in profiles) {
      tasks[[length(tasks) + 1L]] <- build_task(
        row, profile, as.integer(requested$n_bootstraps[[i]]), context
      )
    }
  }
  tasks
}

checkpoint_for_identity <- function(context, manifests, profile, scope, p, s) {
  row <- manifests$stream[
    manifests$stream$scope == scope &
      manifests$stream$possessions_per_bootstrap == p &
      manifests$stream$samples_per_possession == s,
    , drop = FALSE
  ]
  if (nrow(row) != 1L) abort("Could not uniquely locate stream for %s/P=%d/S=%d.", scope, p, s)
  task <- build_task(row, profile, max(FROZEN_CONFIGURATIONS$n_bootstraps[
    FROZEN_CONFIGURATIONS$possessions_per_bootstrap == p &
      FROZEN_CONFIGURATIONS$samples_per_possession == s
  ]), context)
  list(task = task, state = load_checkpoint_state(
    task$checkpoint_path, task$metadata, task$max_bootstraps, TRUE
  ))
}

collect_complete_results <- function(context, manifests, basis) {
  planted <- manifests$planted_profiles$maps
  metric_rows <- list()
  map_bundle <- list()
  diagnostic_rows <- list()
  cache <- new.env(parent = emptyenv())

  for (profile in PROFILES) {
    for (scope in SCOPES) {
      for (config_index in seq_len(nrow(FROZEN_CONFIGURATIONS))) {
        config <- FROZEN_CONFIGURATIONS[config_index, , drop = FALSE]
        cache_key <- paste(profile, scope, config$possessions_per_bootstrap, config$samples_per_possession, sep = "|")
        if (!exists(cache_key, envir = cache, inherits = FALSE)) {
          loaded <- checkpoint_for_identity(
            context, manifests, profile, scope,
            config$possessions_per_bootstrap,
            config$samples_per_possession
          )
          assign(cache_key, loaded, envir = cache)
          for (diag in loaded$state$diagnostics) {
            if (!is.null(diag)) diagnostic_rows[[length(diagnostic_rows) + 1L]] <- diag
          }
        }
        loaded <- get(cache_key, envir = cache, inherits = FALSE)
        b <- as.integer(config$n_bootstraps)
        if (!all(loaded$state$done[seq_len(b)]) || any(!is.finite(loaded$state$maps[, seq_len(b), drop = FALSE]))) {
          return(list(complete = FALSE, metrics = NULL, maps = NULL, diagnostics = list_rows_to_frame(diagnostic_rows)))
        }
        representative <- apply(loaded$state$maps[, seq_len(b), drop = FALSE], 1L, median)
        run_id <- paste(profile, scope, config$config_key, sep = "__")
        similarity <- profile_map_similarity(planted[[profile]], representative)
        metric_rows[[length(metric_rows) + 1L]] <- cbind(
          data.frame(
            run_id = run_id,
            simulation_tag = context$simulation_tag,
            simulated_games = context$simulated_games,
            method = "thin_plate_spline",
            gam_method = context$method,
            basis_id = BASIS_ID,
            basis_k = BASIS_K,
            profile = profile,
            scope = scope,
            config_key = config$config_key,
            n_bootstraps = b,
            possessions_per_bootstrap = as.integer(config$possessions_per_bootstrap),
            samples_per_possession = as.integer(config$samples_per_possession),
            map_estimator = "nested_prefix_median_v1",
            map_status = "ok",
            stringsAsFactors = FALSE
          ),
          similarity
        )
        map_bundle[[run_id]] <- matrix(representative, nrow = NX, ncol = NY, byrow = TRUE)
      }
    }
  }
  metrics <- do.call(rbind, metric_rows)
  rownames(metrics) <- NULL
  expected <- length(PROFILES) * length(SCOPES) * nrow(FROZEN_CONFIGURATIONS)
  if (nrow(metrics) != expected || anyDuplicated(metrics$run_id)) abort("Final spline metrics are incomplete or duplicated.")
  list(
    complete = TRUE,
    metrics = metrics,
    maps = list(
      metadata = list(
        script_version = SCRIPT_VERSION,
        simulation_tag = context$simulation_tag,
        simulated_games = context$simulated_games,
        gam_method = context$method,
        basis_id = BASIS_ID,
        basis_k = BASIS_K,
        shape = c(NX, NY),
        order = "x-major/y-fast"
      ),
      maps = map_bundle
    ),
    diagnostics = list_rows_to_frame(diagnostic_rows)
  )
}

summarize_configuration <- function(rows) {
  data.frame(
    config_key = rows$config_key[[1]],
    n_bootstraps = rows$n_bootstraps[[1]],
    possessions_per_bootstrap = rows$possessions_per_bootstrap[[1]],
    samples_per_possession = rows$samples_per_possession[[1]],
    n_profile_scope_results = length(unique(rows$run_id)),
    n_profiles = length(unique(rows$profile)),
    n_scopes = length(unique(rows$scope)),
    mean_whole_grid_cosine = mean(rows$whole_grid_cosine),
    median_whole_grid_cosine = median(rows$whole_grid_cosine),
    minimum_whole_grid_cosine = min(rows$whole_grid_cosine),
    maximum_whole_grid_cosine = max(rows$whole_grid_cosine),
    mean_lightly_penalized_whole_grid_cosine = mean(rows$lightly_penalized_whole_grid_cosine),
    median_lightly_penalized_whole_grid_cosine = median(rows$lightly_penalized_whole_grid_cosine),
    minimum_lightly_penalized_whole_grid_cosine = min(rows$lightly_penalized_whole_grid_cosine),
    maximum_lightly_penalized_whole_grid_cosine = max(rows$lightly_penalized_whole_grid_cosine),
    mean_moderately_penalized_whole_grid_cosine = mean(rows$moderately_penalized_whole_grid_cosine),
    median_moderately_penalized_whole_grid_cosine = median(rows$moderately_penalized_whole_grid_cosine),
    minimum_moderately_penalized_whole_grid_cosine = min(rows$moderately_penalized_whole_grid_cosine),
    maximum_moderately_penalized_whole_grid_cosine = max(rows$moderately_penalized_whole_grid_cosine),
    mean_support_cosine = mean(rows$support_cosine),
    mean_outside_energy_fraction = mean(rows$outside_energy_fraction),
    mean_recovered_dynamic_range = mean(rows$recovered_dynamic_range),
    expected_profile_scope_results = length(PROFILES) * length(SCOPES),
    coverage_fraction = length(unique(rows$run_id)) / (length(PROFILES) * length(SCOPES)),
    complete_coverage = length(unique(rows$run_id)) == length(PROFILES) * length(SCOPES),
    stringsAsFactors = FALSE
  )
}

build_performance_ranking <- function(metrics) {
  groups <- split(metrics, metrics$config_key)
  ranking <- do.call(rbind, lapply(groups, summarize_configuration))
  rownames(ranking) <- NULL
  order_index <- with(
    ranking,
    order(
      -mean_moderately_penalized_whole_grid_cosine,
      -mean_lightly_penalized_whole_grid_cosine,
      -mean_whole_grid_cosine,
      -mean_support_cosine,
      mean_outside_energy_fraction,
      -minimum_moderately_penalized_whole_grid_cosine,
      n_bootstraps,
      possessions_per_bootstrap,
      samples_per_possession,
      method = "radix"
    )
  )
  ranking <- ranking[order_index, , drop = FALSE]
  ranking$performance_rank <- seq_len(nrow(ranking))
  ranking$ranking_method <- paste(
    "descending mean moderate-penalty whole-grid cosine; then descending mean",
    "light-penalty and raw whole-grid cosine, descending support cosine,",
    "ascending leakage, descending minimum moderate score, then ascending B/P/S"
  )
  ranking <- ranking[, c("performance_rank", "ranking_method", setdiff(names(ranking), c("performance_rank", "ranking_method")))]
  if (nrow(ranking) != nrow(FROZEN_CONFIGURATIONS) || !all(ranking$complete_coverage)) {
    abort("Spline performance ranking does not have complete five-configuration coverage.")
  }
  ranking
}

select_and_build_tasks <- function(options, context, manifests) {
  validate_filter <- function(values, allowed, label) {
    if (is.null(values)) return(allowed)
    unknown <- setdiff(values, allowed)
    if (length(unknown)) abort("Unknown %s value(s): %s.", label, paste(unknown, collapse = ", "))
    values
  }
  profiles <- validate_filter(options$profiles, PROFILES, "profile")
  scopes <- validate_filter(options$scopes, SCOPES, "scope")
  configs <- validate_filter(options$configs, FROZEN_CONFIGURATIONS$config_key, "config")
  stream_keys <- validate_filter(options$stream_keys, manifests$stream$stream_key, "stream-key")
  selected_config <- manifests$config[
    manifests$config$scope %in% scopes &
      manifests$config$config_key %in% configs &
      manifests$config$stream_key %in% stream_keys,
    , drop = FALSE
  ]
  if (!nrow(selected_config)) abort("The filters select no spline work.")
  selected_stream_keys <- unique(selected_config$stream_key)
  validated_stream <- validate_selected_streams(
    context$input_root, manifests, selected_stream_keys
  )
  selected_manifests <- manifests
  selected_manifests$stream <- validated_stream
  list(
    tasks = make_all_tasks(
      context, selected_manifests, profiles = profiles, config = selected_config
    ),
    profiles = profiles,
    config = selected_config,
    stream = validated_stream,
    stream_keys = selected_stream_keys
  )
}

run_tasks <- function(tasks, basis, requested_workers) {
  if (!length(tasks)) return(list())
  physical <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (is.na(physical) || physical < 1L) physical <- 1L
  available <- max(1L, physical - 1L)
  workers <- min(as.integer(requested_workers), MAX_PSOCK_WORKERS, available, length(tasks))
  cat(sprintf("Using %d PSOCK worker(s) for %d logical stream task(s).\n", workers, length(tasks)))
  if (workers == 1L) return(lapply(tasks, process_stream_task, basis = basis))

  cluster <- parallel::makeCluster(workers, type = "PSOCK")
  on.exit(parallel::stopCluster(cluster), add = TRUE)
  parallel::clusterEvalQ(cluster, {
    Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    if (!requireNamespace("mgcv", quietly = TRUE)) stop("mgcv is unavailable on a worker.")
    NULL
  })
  exports <- c(
    "SCRIPT_VERSION", "N_CELLS", "checkpoint_backup_path", "recover_checkpoint_path",
    "atomic_save_rds", "metadata_signature", "new_checkpoint_state", "load_checkpoint_state",
    "abort", "collapse_messages", "read_u8_slot", "format_numbers",
    "fit_spline_slot", "process_stream_task"
  )
  parallel::clusterExport(cluster, exports, envir = environment())
  parallel::parLapplyLB(cluster, tasks, process_stream_task, basis = basis)
}

main <- function() {
  options <- parse_cli(commandArgs(trailingOnly = TRUE))
  if (options$help) {
    print_help()
    return(invisible(0L))
  }
  if (!requireNamespace("mgcv", quietly = TRUE)) abort("The mgcv package is required.")
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")

  input_root <- normalizePath(options$input_root, winslash = "/", mustWork = TRUE)
  output_root <- normalizePath(options$output_root, winslash = "/", mustWork = FALSE)
  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
  output_root <- normalizePath(output_root, winslash = "/", mustWork = TRUE)
  raw_config <- read_manifest(file.path(input_root, "config_manifest.csv"), "config manifest")
  raw_stream <- read_manifest(file.path(input_root, "stream_manifest.csv"), "stream manifest")
  manifests <- validate_manifests(input_root, raw_config, raw_stream)
  basis <- build_basis()
  context <- list(
    input_root = input_root,
    output_root = output_root,
    simulation_tag = manifests$simulation_tag,
    simulated_games = manifests$simulated_games,
    method = options$method,
    resume = options$resume,
    input_metadata_sha256 = manifests$global_metadata$sha256,
    planted_profiles_sha256 = manifests$planted_profiles$sha256
  )
  selection <- select_and_build_tasks(options, context, manifests)
  tasks <- selection$tasks
  cat(sprintf(
    "Validated %d-game inputs; method=%s, basis=%s, selected tasks=%d.\n",
    context$simulated_games, context$method, BASIS_ID, length(tasks)
  ))
  if (options$dry_run) {
    preview <- do.call(rbind, lapply(tasks, function(task) data.frame(
      profile = task$profile,
      scope = task$scope,
      stream_key = task$stream_key,
      required_bootstraps = task$required_bootstraps,
      checkpoint_path = task$checkpoint_path,
      stringsAsFactors = FALSE
    )))
    print(preview, row.names = FALSE)
    return(invisible(0L))
  }

  task_results <- run_tasks(tasks, basis, options$workers)
  failures <- Filter(function(result) !isTRUE(result$complete), task_results)
  if (length(failures)) {
    labels <- vapply(failures, function(result) {
      sprintf("%s/%s slots=%s", result$profile, result$task, paste(result$failed_slots, collapse = ","))
    }, character(1))
    abort("Spline task failures remain: %s", paste(labels, collapse = " | "))
  }

  if (options$finalize) {
    if (setequal(selection$stream_keys, manifests$stream$stream_key)) {
      final_manifests <- manifests
      final_manifests$stream <- selection$stream
      final <- collect_complete_results(context, final_manifests, basis)
    } else if (all(normalize_token(manifests$stream$status) == "valid")) {
      final_manifests <- manifests
      final_manifests$stream <- validate_selected_streams(
        input_root, manifests, manifests$stream$stream_key
      )
      final <- collect_complete_results(context, final_manifests, basis)
    } else {
      final <- list(complete = FALSE, metrics = NULL, maps = NULL, diagnostics = data.frame())
    }
    if (!final$complete) {
      cat("Selected tasks completed, but the full 100-row study is not complete; canonical outputs were not published.\n")
    } else {
      ranking <- build_performance_ranking(final$metrics)
      atomic_write_csv(final$metrics, file.path(output_root, "representative_map_similarity.csv"))
      atomic_write_csv(ranking, file.path(output_root, "performance_ranking_overall.csv"))
      atomic_write_csv(final$diagnostics, file.path(output_root, "spline_slot_diagnostics.csv"))
      atomic_save_rds(final$maps, file.path(output_root, "representative_maps.rds"))
      cat(sprintf("Published %d matched spline metric rows and %d ranking rows under %s.\n", nrow(final$metrics), nrow(ranking), output_root))
    }
  }
  invisible(0L)
}

if (sys.nframe() == 0L) {
  status <- tryCatch(main(), error = function(e) {
    message("ERROR: ", conditionMessage(e))
    1L
  })
  if (!identical(status, 0L)) quit(save = "no", status = status, runLast = FALSE)
}
