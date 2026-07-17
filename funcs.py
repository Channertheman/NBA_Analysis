import pandas as pd
import cvxpy as cp
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import lil_matrix
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import tqdm
from tqdm import trange

# ============================================================
# 1) Core λ-max / QUT machinery (infinity-norm program + null sims)
# ============================================================

def _format_grid_variable_name(index, grid_height=10):
    return f"x_bin={index // grid_height}, y_bin={index % grid_height}"


def diagnose_infinity_norm_feasibility(
    y,
    X,
    D,
    variable_names=None,
    top_n=15,
    grid_height=10,
):
    """
    Diagnose whether grad_h lies in the column space of D.T.

    Without projecting grad_h, the constraint D.T @ w == grad_h is feasible
    only when grad_h is exactly representable by D.T. The residuals below show
    which variables are farthest from the nearest feasible gradient.
    """
    X0 = X @ np.ones((X.shape[1], 1))
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs')
    clf.fit(X0, y)
    beta0 = clf.coef_.flatten()

    u0 = beta0[0] * np.ones(X.shape[1])
    p = 1 / (1 + np.exp(-X @ u0))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    grad_h = np.asarray(X.T @ (y - p)).reshape(-1)

    A = np.asarray(D.T, dtype=float)
    nearest_w, *_ = np.linalg.lstsq(A, grad_h, rcond=None)
    nearest_grad_h = A @ nearest_w
    residual = grad_h - nearest_grad_h
    abs_residual = np.abs(residual)
    order = np.argsort(abs_residual)[::-1][:top_n]

    if variable_names is None:
        variable_names = [
            _format_grid_variable_name(i, grid_height=grid_height)
            for i in range(X.shape[1])
        ]

    top_variables = pd.DataFrame({
        "variable_index": order,
        "variable": [variable_names[i] for i in order],
        "grad_h": grad_h[order],
        "nearest_feasible_grad_h": nearest_grad_h[order],
        "residual": residual[order],
        "abs_residual": abs_residual[order],
    })

    return {
        "top_variables": top_variables,
        "grad_h": grad_h,
        "nearest_feasible_grad_h": nearest_grad_h,
        "residual": residual,
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_linf": float(np.max(abs_residual)),
        "rank_D_transpose": int(np.linalg.matrix_rank(A)),
        "n_variables": int(A.shape[0]),
        "n_constraints": int(A.shape[1]),
    }


def solve_infinity_norm(
    y,
    X,
    D,
    project_grad_h=False,
    debug_infeasible=False,
    variable_names=None,
    infeasible_top_n=15,
):
    m = D.shape[0]

    # Step 1: Fit logistic regression with constant input
    try:
        X0 = X @ np.ones((X.shape[1], 1))
        clf = LogisticRegression(fit_intercept=False, solver='lbfgs')
        clf.fit(X0, y)
        beta0 = clf.coef_.flatten()
    except Exception as e:
        raise ValueError(f"Logistic regression failed: {e}")

    # Step 2: Compute predicted probabilities
    u0 = beta0[0] * np.ones(X.shape[1])
    p = 1 / (1 + np.exp(-X @ u0))
    p = np.clip(p, 1e-6, 1 - 1e-6)

    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        raise ValueError("Invalid probabilities p (NaN or Inf)")

    # Step 3: Compute gradient
    grad_h = X.T @ (y - p)

    if np.any(np.isnan(grad_h)) or np.any(np.isinf(grad_h)):
        raise ValueError("Invalid grad_h (NaN or Inf)")

    # Optional: project grad_h to column space of Dᵗ
    if project_grad_h:
        try:
            P = D.T @ pinv(D @ D.T) @ D
            grad_h = grad_h.reshape(-1, 1)
            grad_h = (P @ grad_h).flatten()

        except Exception as e:
            raise ValueError(f"Projection failed: {e}")

    # Step 4: Solve the constrained optimization
    w = cp.Variable(m)
    constraints = [D.T @ w == grad_h]
    objective = cp.Minimize(cp.norm(w, "inf"))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    # Step 5: Check solver status
    if prob.status != "optimal":
        if debug_infeasible:
            diagnostics = diagnose_infinity_norm_feasibility(
                y,
                X,
                D,
                variable_names=variable_names,
                top_n=infeasible_top_n,
            )
            print("Optimization infeasibility diagnostics")
            print(f"Status: {prob.status}")
            print(
                "grad_h distance from col(D.T): "
                f"L2={diagnostics['residual_l2']:.6g}, "
                f"Linf={diagnostics['residual_linf']:.6g}"
            )
            print(
                "rank(D.T)="
                f"{diagnostics['rank_D_transpose']} / "
                f"{diagnostics['n_variables']} variables"
            )
            print(diagnostics["top_variables"].to_string(index=False))
        raise ValueError(f"Optimization failed. Status: {prob.status}")

    return w.value, prob.value


def lambda_qut(
    X,
    y,
    D_TV,
    MC=1000,
    project_grad_h=True,
    debug_infeasible=False,
    variable_names=None,
    infeasible_top_n=15,
    stop_on_infeasible=False,
):
    X0 = X @ np.ones((X.shape[1], 1))
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs')
    clf.fit(X0, y)
    x0 = clf.coef_.flatten()
    mu0 = X0 @ x0
    p0 = 1 / (1 + np.exp(-mu0))
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)

    Lambdas = []
    attempts = 0
    null_debug_used = False

    print("Simulating null responses...")
    while len(Lambdas) < MC and attempts < MC * 20:
        attempts += 1
        y_sim = binom.rvs(1, p0)

        if np.all(y_sim == 0) or np.all(y_sim == 1):
            print(f"Skipping degenerate simulation (all {int(y_sim[0])})")
            continue

        try:
            _, lambda_val = solve_infinity_norm(
                y_sim,
                X,
                D_TV,
                project_grad_h=project_grad_h,
                debug_infeasible=debug_infeasible and not null_debug_used,
                variable_names=variable_names,
                infeasible_top_n=infeasible_top_n,
            )
            if np.isfinite(lambda_val):
                Lambdas.append(lambda_val)
                if len(Lambdas) % 100 == 0:
                    print(f"Accepted {len(Lambdas)} / {MC} simulations")
            else:
                print("Skipping non-finite lambda")
        except Exception as e:
            if debug_infeasible and not null_debug_used:
                null_debug_used = True
            if stop_on_infeasible:
                raise RuntimeError(
                    "QUT null simulation failed while solving the projection-off "
                    "infinity-norm problem. See infeasibility diagnostics above."
                ) from e
            print(f"Skipping simulation due to error: {e}")

    if len(Lambdas) == 0:
        raise RuntimeError("All simulations failed. Check your data or D matrix.")

    lambda_qut_val = np.quantile(Lambdas, 0.95)

    try:
        _, lambda_max = solve_infinity_norm(
            y,
            X,
            D_TV,
            project_grad_h=project_grad_h,
            debug_infeasible=debug_infeasible,
            variable_names=variable_names,
            infeasible_top_n=infeasible_top_n,
        )
    except Exception as e:
        raise RuntimeError(f"Optimization on real data failed: {e}")

    print(f"λ_qut (95th percentile null): {lambda_qut_val:.4f}")
    print(f"λ_max (data): {lambda_max:.4f}")
    return lambda_qut_val, lambda_max, Lambdas


# ============================================================
# 2) Grid / design-matrix builders (X, y, D_TV)
# ============================================================

def grid(df):
    grid_width = 20
    grid_height = 10
    num_bins = grid_width * grid_height
    
    # Step 1: Map each (x_bin, y_bin) to index
    bin_to_index = {(x, y): x * grid_height + y for x in range(grid_width) for y in range(grid_height)}
    
    # Step 2: Prepare possession-level feature matrix X and outcome vector y
    possessions = df['possession_number'].unique()
    n_possessions = len(possessions)
    X = lil_matrix((n_possessions, num_bins), dtype=np.float32)
    y = np.array([group['scored'].iloc[0] for _, group in df.groupby('possession_number')], dtype=np.int8)
    possession_to_index = {p: i for i, p in enumerate(possessions)}
    
    for pid, group in df.groupby('possession_number'):
        i = possession_to_index[pid]
        visited_bins = set(zip(group['x_bin'], group['y_bin']))
        for xb, yb in visited_bins:
            if (xb, yb) in bin_to_index:
                X[i, bin_to_index[(xb, yb)]] = 1
        y[i] = group['scored'].iloc[0]
    
    rows = []
    for xx in range(grid_width - 1):
        for yy in range(grid_height):
            i = xx * grid_height + yy
            row = np.zeros(num_bins)
            row[i] = -1
            row[i + grid_height] = 1
            rows.append(row)
    
    for xx in range(grid_width):
        for yy in range(grid_height - 1):
            i = xx * grid_height + yy
            row = np.zeros(num_bins)
            row[i] = -1
            row[i + 1] = 1
            rows.append(row)
    D_TV = np.vstack(rows)
    return X, y, D_TV


# ============================================================
# 3) TV-logistic model solvers (single fit)
# ============================================================

def tv_logistic_regression_heatmap(df, lambda_tv=1.0, grid_width=20, grid_height=10, court_image_path="court.jpg", plot = False):
    num_bins = grid_width * grid_height

    # Map (x_bin, y_bin) to linear index
    bin_to_index = {(x, y): x * grid_height + y for x in range(grid_width) for y in range(grid_height)}

    # Unique possessions
    possessions = df['possession_number'].unique()
    n_possessions = len(possessions)
    X = lil_matrix((n_possessions, num_bins), dtype=np.float32)
    y = np.array([group['scored'].iloc[0] for _, group in df.groupby('possession_number')], dtype=np.int8)
    possession_to_index = {p: i for i, p in enumerate(possessions)}

    for pid, group in df.groupby('possession_number'):
        i = possession_to_index[pid]
        visited_bins = set(zip(group['x_bin'], group['y_bin']))
        for xb, yb in visited_bins:
            if (xb, yb) in bin_to_index:
                X[i, bin_to_index[(xb, yb)]] = 1
        y[i] = group['scored'].iloc[0]

    # Define CVXPY variables
    beta_grid = cp.Variable((grid_width, grid_height))
    beta_flat = cp.vec(beta_grid, order="C")
    logits = X @ beta_flat
    log_likelihood = cp.sum(cp.multiply(y, logits) - cp.logistic(logits))

    # TV penalty terms
    tv_terms = []
    for x in range(grid_width):
        for y in range(grid_height):
            if x + 1 < grid_width:
                tv_terms.append(cp.abs(beta_grid[x, y] - beta_grid[x + 1, y]))
            if y + 1 < grid_height:
                tv_terms.append(cp.abs(beta_grid[x, y] - beta_grid[x, y + 1]))

    # Regularized objective
    constraints = [beta_grid >= -10, beta_grid <= 10]
    tv_penalty = cp.sum(tv_terms)
    objective = cp.Maximize(log_likelihood - lambda_tv * tv_penalty)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem
    try:
        result = problem.solve(solver=cp.ECOS, verbose=False)
    except:
        print("ECOS failed, trying SCS...")
        result = problem.solve(solver=cp.SCS, verbose=False)

    if beta_grid.value is None:
        raise RuntimeError("Optimization failed. Check data integrity or constraints.")

    beta_est = beta_grid.value

    # Visualization
    if plot == True:
        court_img = mpimg.imread(court_image_path)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(court_img, extent=[0, grid_width, 0, grid_height], aspect='auto')
    
        heatmap = ax.imshow(beta_est.T, origin="lower", extent=[0, grid_width, 0, grid_height],
                            cmap="viridis", alpha=0.75)

        ax.set_title(f"Estimated Scoring Threat Map, Lambda = {lambda_tv}")
        ax.set_xlabel("x_bin")
        ax.set_ylabel("y_bin")
        plt.colorbar(heatmap, ax=ax, label="β̂ value")
        plt.tight_layout()
        plt.show()

    return beta_est


def solve_tv_logistic_regression(df, lambda_tv=1.0, grid_width=20, grid_height=10):
    num_bins = grid_width * grid_height
    bin_to_index = {(x, y): x * grid_height + y for x in range(grid_width) for y in range(grid_height)}
    possessions = df['possession_number'].unique()
    n_possessions = len(possessions)
    X = lil_matrix((n_possessions, num_bins), dtype=np.float32)
    y = np.array([group['scored'].iloc[0] for _, group in df.groupby('possession_number')], dtype=np.int8)
    possession_to_index = {p: i for i, p in enumerate(possessions)}

    for pid, group in df.groupby('possession_number'):
        i = possession_to_index[pid]
        visited_bins = set(zip(group['x_bin'], group['y_bin']))
        for xb, yb in visited_bins:
            if (xb, yb) in bin_to_index:
                X[i, bin_to_index[(xb, yb)]] = 1
        y[i] = group['scored'].iloc[0]

    beta_grid = cp.Variable((grid_width, grid_height))
    beta_flat = cp.vec(beta_grid, order="C")
    logits = X @ beta_flat
    log_likelihood = cp.sum(cp.multiply(y, logits) - cp.logistic(logits))

    tv_terms = []
    for x in range(grid_width):
        for y in range(grid_height):
            if x + 1 < grid_width:
                tv_terms.append(cp.abs(beta_grid[x, y] - beta_grid[x + 1, y]))
            if y + 1 < grid_height:
                tv_terms.append(cp.abs(beta_grid[x, y] - beta_grid[x, y + 1]))

    constraints = [beta_grid >= -10, beta_grid <= 10]
    tv_penalty = cp.sum(tv_terms)
    objective = cp.Maximize(log_likelihood - lambda_tv * tv_penalty)
    problem = cp.Problem(objective, constraints)

    try:
        result = problem.solve(solver=cp.ECOS, verbose=False)
    except:
        print("ECOS failed, trying SCS...")
        result = problem.solve(solver=cp.SCS, verbose=False)

    if beta_grid.value is None:
        raise RuntimeError("Optimization failed.")

    return beta_grid.value  # No plotting


# ============================================================
# 4) Bootstrapping: dataset generation + repeated model fits
# ============================================================

def generate_bootstrapped_datasets(
    original_df,
    n_bootstraps=100,
    n_possessions_per_bootstrap=1000,
    samples_per_possession=100,
    min_samples_required=100,
    seed=42
):
    """
    Generate B bootstrapped datasets using possession-level resampling.

    Parameters:
        original_df (pd.DataFrame): Raw dataset with columns ['possession_number', 'x', 'y', 'scored', ...]
        n_bootstraps (int): Number of bootstrapped datasets to generate
        n_possessions_per_bootstrap (int): Number of possessions in each bootstrapped dataset
        samples_per_possession (int): How many samples to draw per possession
        min_samples_required (int): Minimum number of rows a possession must have to be eligible for sampling
        seed (int): RNG seed for reproducibility

    Returns:
        List[pd.DataFrame]: List of bootstrapped DataFrames
    """
    rng = np.random.default_rng(seed)
    
    # Filter out sparse possessions
    grouped = list(original_df.groupby("possession_number"))
    eligible_possessions = [g for _, g in grouped if len(g) >= min_samples_required]
    total_eligible = len(eligible_possessions)

    if total_eligible == 0:
        raise ValueError("No possessions meet the minimum sample requirement.")

    bootstrapped_datasets = []

    for b in tqdm.tqdm(range(n_bootstraps), desc="Generating Bootstrapped Datasets"):
        new_rows = []
        new_possession_ids = range(100000 + b * n_possessions_per_bootstrap,
                                   100000 + (b + 1) * n_possessions_per_bootstrap)

        for new_pid in new_possession_ids:
            real_group = eligible_possessions[rng.integers(total_eligible)]
            sampled_rows = real_group.sample(
                n=samples_per_possession,
                replace=True,
                random_state=rng.integers(1e6)
            ).copy()
            sampled_rows["possession_number"] = new_pid
            new_rows.append(sampled_rows)

        df_boot = pd.concat(new_rows, ignore_index=True)
        bootstrapped_datasets.append(df_boot)

    return bootstrapped_datasets


def _fit_unpenalized_qut_null(
    X,
    y,
    solver="lbfgs",
    tol=1e-12,
    max_iter=1000,
    probability_clip=1e-6,
):
    """Fit the common-coefficient logistic null used by the QUT gradient."""
    y_array = np.asarray(y).reshape(-1)
    X0 = np.asarray(X @ np.ones((X.shape[1], 1), dtype=float), dtype=float)

    if X0.shape[0] != y_array.size:
        raise ValueError("X and y have incompatible row counts.")
    if not np.isfinite(X0).all() or not np.isfinite(y_array).all():
        raise ValueError("The QUT null fit received non-finite data.")
    if np.unique(y_array).size < 2:
        raise ValueError("The QUT null fit requires both outcome classes.")

    model = LogisticRegression(
        fit_intercept=False,
        solver=solver,
        penalty=None,
        tol=tol,
        max_iter=max_iter,
    )
    model.fit(X0, y_array)
    beta0 = float(model.coef_.reshape(-1)[0])
    linear_predictor = np.asarray(X @ np.full(X.shape[1], beta0)).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-linear_predictor))
    probabilities = np.clip(
        probabilities,
        probability_clip,
        1.0 - probability_clip,
    )
    likelihood_score = float(X0.reshape(-1) @ (y_array - probabilities))
    return probabilities, beta0, likelihood_score


def analyze_bootstrap_qut_feasibility(
    boot_datasets,
    D_TV,
    keep_mask=None,
    variable_indices=None,
    current_slack_tol=1e-7,
    grid_builder=None,
    grid_height=10,
    coverage_quantiles=(0.50, 0.90, 0.95, 0.99, 1.00),
    null_solver="lbfgs",
    null_tol=1e-12,
    null_max_iter=1000,
    probability_clip=1e-6,
):
    """
    Measure projection-off QUT feasibility over bootstrap datasets.

    For each bootstrap this function fits the unpenalized common-coefficient
    logistic null, computes ``g = X.T @ (y - p0)``, and measures the residual
    between ``g`` and its nearest representable value in ``col(D_TV.T)``.

    For a TV incidence matrix, the maximum absolute projection residual is the
    minimum component-wise uniform slack required to satisfy
    ``D_TV.T @ w + slack == g``. Thus ``minimum_slack_linf`` is the decisive
    statistic for comparing a bootstrap with ``current_slack_tol``.

    Parameters
    ----------
    boot_datasets : sequence of pandas.DataFrame
        Possession-bootstrap datasets accepted by ``grid_builder``.
    D_TV : array-like, shape (n_edges, n_retained_sectors)
        Full or reduced TV incidence matrix used by the QUT optimization.
    keep_mask : array-like of bool, optional
        Mask selecting retained columns from each full bootstrap design matrix.
        If omitted, every design-matrix column is retained.
    variable_indices : array-like of int, optional
        Original full-grid indices for the retained sectors. Inferred from
        ``keep_mask`` when possible.
    current_slack_tol : float, default 1e-7
        Slack bound whose empirical bootstrap coverage should be reported.
    grid_builder : callable, optional
        Callable returning ``(X, y, D)`` from a bootstrap dataframe. Defaults
        to this module's ``grid`` function.

    Returns
    -------
    dict
        Detailed sector/bootstrap residuals, per-bootstrap and per-sector
        summaries, pooled distribution statistics, slack guidance, failures,
        and current-slack bootstrap coverage.
    """
    if grid_builder is None:
        grid_builder = grid
    if current_slack_tol is None or current_slack_tol < 0:
        raise ValueError("current_slack_tol must be a non-negative number.")

    D_reduced = np.asarray(D_TV, dtype=float)
    if D_reduced.ndim != 2:
        raise ValueError("D_TV must be a two-dimensional matrix.")

    # The closed-form minimum-slack interpretation relies on the rows being TV
    # incidence edges: two nonzero entries with a zero row sum.
    if D_reduced.shape[0]:
        nonzero_per_row = np.count_nonzero(D_reduced, axis=1)
        is_tv_incidence = bool(
            np.all(nonzero_per_row == 2)
            and np.allclose(D_reduced.sum(axis=1), 0.0)
        )
        if not is_tv_incidence:
            raise ValueError(
                "D_TV must be a TV incidence matrix with two opposite entries per edge."
            )

    keep_mask_array = None if keep_mask is None else np.asarray(keep_mask, dtype=bool)
    if keep_mask_array is not None:
        inferred_indices = np.flatnonzero(keep_mask_array)
        if inferred_indices.size != D_reduced.shape[1]:
            raise ValueError(
                "The retained-sector mask and D_TV have incompatible dimensions."
            )
    else:
        inferred_indices = np.arange(D_reduced.shape[1], dtype=int)

    if variable_indices is None:
        variable_indices_array = inferred_indices
    else:
        variable_indices_array = np.asarray(variable_indices, dtype=int)
        if variable_indices_array.size != D_reduced.shape[1]:
            raise ValueError(
                "variable_indices must contain one index per D_TV column."
            )

    D_transpose = D_reduced.T
    feasible_projection = D_transpose @ np.linalg.pinv(D_transpose)
    bootstrap_rows = []
    sector_frames = []

    for bootstrap_index, df_boot in enumerate(
        tqdm.tqdm(boot_datasets, desc="Calculating bootstrap QUT feasibility")
    ):
        try:
            X_boot, y_boot, _ = grid_builder(df_boot)
            if keep_mask_array is None:
                X_reduced = X_boot
            else:
                if X_boot.shape[1] != keep_mask_array.size:
                    raise ValueError(
                        f"Bootstrap has {X_boot.shape[1]} sectors; "
                        f"expected {keep_mask_array.size}."
                    )
                X_reduced = X_boot[:, keep_mask_array]

            if X_reduced.shape[1] != D_reduced.shape[1]:
                raise ValueError(
                    "The retained bootstrap design and D_TV have incompatible dimensions."
                )

            probabilities, beta0, null_score = _fit_unpenalized_qut_null(
                X_reduced,
                y_boot,
                solver=null_solver,
                tol=null_tol,
                max_iter=null_max_iter,
                probability_clip=probability_clip,
            )
            grad_h = np.asarray(
                X_reduced.T @ (np.asarray(y_boot).reshape(-1) - probabilities)
            ).reshape(-1)
            nearest_feasible_grad_h = feasible_projection @ grad_h
            residual = grad_h - nearest_feasible_grad_h
            abs_residual = np.abs(residual)
            minimum_slack_linf = float(np.max(abs_residual))

            sector_frames.append(pd.DataFrame({
                "bootstrap_index": bootstrap_index,
                "reduced_variable_index": np.arange(residual.size),
                "variable_index": variable_indices_array,
                "x_bin": variable_indices_array // grid_height,
                "y_bin": variable_indices_array % grid_height,
                "grad_h": grad_h,
                "nearest_feasible_grad_h": nearest_feasible_grad_h,
                "residual": residual,
                "abs_residual": abs_residual,
            }))
            bootstrap_rows.append({
                "bootstrap_index": bootstrap_index,
                "n_possessions": int(X_reduced.shape[0]),
                "scored_rate": float(np.mean(y_boot)),
                "null_beta0": beta0,
                "null_likelihood_score": null_score,
                "mean_residual": float(np.mean(residual)),
                "variance_residual": float(np.var(residual, ddof=1)),
                "std_residual": float(np.std(residual, ddof=1)),
                "min_residual": float(np.min(residual)),
                "max_residual": float(np.max(residual)),
                "range_residual": float(np.ptp(residual)),
                "mean_abs_residual": float(np.mean(abs_residual)),
                "median_abs_residual": float(np.median(abs_residual)),
                "p90_abs_residual": float(np.quantile(abs_residual, 0.90)),
                "p95_abs_residual": float(np.quantile(abs_residual, 0.95)),
                "p99_abs_residual": float(np.quantile(abs_residual, 0.99)),
                "max_abs_residual": minimum_slack_linf,
                "minimum_slack_linf": minimum_slack_linf,
                "slack_shortfall": max(minimum_slack_linf - current_slack_tol, 0.0),
                "covered_by_current_slack": minimum_slack_linf <= current_slack_tol,
                "rmse_residual": float(np.sqrt(np.mean(residual ** 2))),
                "l2_residual": float(np.linalg.norm(residual)),
                "error": None,
            })
        except Exception as exc:
            bootstrap_rows.append({
                "bootstrap_index": bootstrap_index,
                "error": str(exc),
            })

    per_bootstrap = pd.DataFrame(bootstrap_rows)
    if not sector_frames:
        raise RuntimeError("QUT feasibility analysis failed for every bootstrap dataset.")

    by_sector_bootstrap = pd.concat(sector_frames, ignore_index=True)
    successful = per_bootstrap[per_bootstrap["error"].isna()].copy()

    def describe_distribution(label, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return {
            "residual_measure": label,
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=1)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.ptp(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }

    distribution_summary = pd.DataFrame([
        describe_distribution(
            "signed residual (all retained sectors and bootstraps)",
            by_sector_bootstrap["residual"],
        ),
        describe_distribution(
            "absolute residual (all retained sectors and bootstraps)",
            by_sector_bootstrap["abs_residual"],
        ),
        describe_distribution(
            "minimum L_inf slack required per bootstrap",
            successful["minimum_slack_linf"],
        ),
    ]).set_index("residual_measure")

    sector_summary = (
        by_sector_bootstrap
        .groupby(["variable_index", "x_bin", "y_bin"], as_index=False)
        .agg(
            mean_residual=("residual", "mean"),
            variance_residual=("residual", "var"),
            min_residual=("residual", "min"),
            max_residual=("residual", "max"),
            mean_abs_residual=("abs_residual", "mean"),
            p95_abs_residual=("abs_residual", lambda values: values.quantile(0.95)),
            max_abs_residual=("abs_residual", "max"),
            proportion_over_current_slack=(
                "abs_residual",
                lambda values: (values > current_slack_tol).mean(),
            ),
        )
    )
    sector_summary["range_residual"] = (
        sector_summary["max_residual"] - sector_summary["min_residual"]
    )
    sector_summary = sector_summary.sort_values("p95_abs_residual", ascending=False)

    coverage_quantiles = np.asarray(coverage_quantiles, dtype=float)
    if (
        coverage_quantiles.ndim != 1
        or coverage_quantiles.size == 0
        or np.any((coverage_quantiles < 0) | (coverage_quantiles > 1))
    ):
        raise ValueError("coverage_quantiles must contain probabilities between 0 and 1.")
    slack_guidance = pd.DataFrame({
        "bootstrap_coverage_target": coverage_quantiles,
        "suggested_slack_tol": np.quantile(
            successful["minimum_slack_linf"], coverage_quantiles
        ),
    })
    slack_guidance["multiple_of_current_slack_tol"] = (
        slack_guidance["suggested_slack_tol"] / current_slack_tol
        if current_slack_tol > 0
        else np.nan
    )

    return {
        "residual_definition": "grad_h - projection_to_col(D_TV.T)",
        "by_sector_bootstrap": by_sector_bootstrap,
        "per_bootstrap": per_bootstrap,
        "distribution_summary": distribution_summary,
        "sector_summary": sector_summary,
        "slack_guidance": slack_guidance,
        "current_slack_tol": float(current_slack_tol),
        "current_slack_bootstrap_coverage": float(
            successful["covered_by_current_slack"].mean()
        ),
        "maximum_required_slack": float(successful["minimum_slack_linf"].max()),
        "maximum_slack_multiple": (
            float(successful["minimum_slack_linf"].max() / current_slack_tol)
            if current_slack_tol > 0
            else np.nan
        ),
        "failed_bootstraps": per_bootstrap[per_bootstrap["error"].notna()].copy(),
    }


def plot_bootstrap_qut_feasibility(
    residual_results,
    current_slack_tol=None,
    absolute_residual_bins=50,
    minimum_slack_bins=40,
):
    """Plot sector residuals and per-bootstrap minimum QUT slack requirements."""
    if current_slack_tol is None:
        current_slack_tol = residual_results["current_slack_tol"]

    absolute_residuals = residual_results["by_sector_bootstrap"]["abs_residual"]
    minimum_slack = (
        residual_results["per_bootstrap"]
        .loc[lambda frame: frame["error"].isna(), "minimum_slack_linf"]
    )
    p95_slack = float(minimum_slack.quantile(0.95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    axes[0].hist(
        absolute_residuals,
        bins=absolute_residual_bins,
        color="steelblue",
        alpha=0.85,
    )
    axes[0].axvline(
        current_slack_tol,
        color="black",
        linestyle="--",
        label=f"current slack={current_slack_tol:.3g}",
    )
    axes[0].set_title("Absolute sector residuals across bootstraps")
    axes[0].set_xlabel("abs(feasibility residual)")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(
        minimum_slack,
        bins=minimum_slack_bins,
        color="darkorange",
        alpha=0.85,
    )
    axes[1].axvline(
        current_slack_tol,
        color="black",
        linestyle="--",
        label="current slack",
    )
    axes[1].axvline(
        p95_slack,
        color="crimson",
        linestyle=":",
        label=f"95% coverage={p95_slack:.3g}",
    )
    axes[1].set_title("Minimum uniform slack required by bootstrap")
    axes[1].set_xlabel("minimum required L_inf slack")
    axes[1].set_ylabel("bootstrap count")
    axes[1].legend()
    return fig, axes


def bootstrap_tv_logistic(df, B=100, lambda_tv=1.0, seed=42):
    np.random.seed(seed)
    possessions = df['possession_number'].unique()
    grid_width, grid_height = 20, 10
    beta_bootstrap = np.zeros((B, grid_width, grid_height))

    for b in tqdm.tqdm(range(B), desc="Bootstrapping"):
        sample_ids = np.random.choice(possessions, size=len(possessions), replace=True)
        df_sample = df[df['possession_number'].isin(sample_ids)]
        
        try:
            beta_b = solve_tv_logistic_regression(df_sample, lambda_tv=lambda_tv,
                                                    grid_width=grid_width, grid_height=grid_height)
                                                    #court_image_path="court.jpg")
            beta_bootstrap[b] = beta_b
        except RuntimeError as e:
            print(f"Bootstrap {b} failed: {e}")
            continue

    return beta_bootstrap


def generate_beta_samples_from_bootstraps(boot_datasets, lambda_tv=1.0):
    beta_list = []

    for df_b in tqdm.tqdm(boot_datasets, desc="Solving TV-Logistic"):
        try:
            beta_b = solve_tv_logistic_regression(df_b, lambda_tv=lambda_tv)
            beta_list.append(beta_b)
        except RuntimeError as e:
            print("Skipped due to optimization failure:", e)
            continue

    return np.stack(beta_list)


# ============================================================
# 5) λ-max over bootstraps + λ-max summary stats
# ============================================================

def compute_lambda_max_from_bootstraps(boot_datasets, D_TV, solve_lambda_func, project_grad_h=True):
    lambda_max_list = []

    for df_b in tqdm.tqdm(boot_datasets, desc="Computing λ_max for bootstraps"):
        # Build X, y from each bootstrapped dataset
        possessions = df_b['possession_number'].unique()
        grid_width, grid_height = 20, 10
        num_bins = grid_width * grid_height

        X = lil_matrix((len(possessions), num_bins), dtype=np.float32)
        y = np.array([group['scored'].iloc[0] for _, group in df_b.groupby('possession_number')], dtype=np.int8)
        possession_to_index = {p: i for i, p in enumerate(possessions)}
        bin_to_index = {(x, y): x * grid_height + y for x in range(grid_width) for y in range(grid_height)}

        for pid, group in df_b.groupby('possession_number'):
            i = possession_to_index[pid]
            visited_bins = set(zip(group['x_bin'], group['y_bin']))
            for xb, yb in visited_bins:
                if (xb, yb) in bin_to_index:
                    X[i, bin_to_index[(xb, yb)]] = 1

        try:
            _, lambda_max_b = solve_lambda_func(y, X, D_TV, project_grad_h=project_grad_h)
            if np.isfinite(lambda_max_b):
                lambda_max_list.append(lambda_max_b)
        except:
            continue

    return np.array(lambda_max_list)


def summarize_lambda_max_analysis(
    lambda_max_obs,
    lambda_max_null,
    lambda_max_bootstrap,
    plot=True
):
    """
    Summarize the λ_max statistical test including CI, p-value, and QUT.

    Parameters:
    - lambda_max_obs (float): Observed λ_max from real or bootstrapped data
    - lambda_max_null (array-like): λ_max values simulated under the null
    - lambda_max_bootstrap (array-like): λ_max values from bootstrapped real data
    - plot (bool): Whether to plot the distributions

    Returns:
    - dict with keys: 'lambda_qut', 'lambda_max_obs', 'ci', 'p_value'
    """

    lambda_max_null = np.asarray(lambda_max_null)
    lambda_max_bootstrap = np.asarray(lambda_max_bootstrap)

    lambda_qut = np.percentile(lambda_max_null, 95)
    ci_lower, ci_upper = np.quantile(lambda_max_bootstrap, [0.025, 0.975])
    p_val = np.mean(lambda_max_null >= lambda_max_obs)

    print(f"λ_qut (95th percentile null): {lambda_qut:.4f}")
    print(f"λ_max (observed): {lambda_max_obs:.4f}")
    print(f"95% CI for λ_max (bootstrap): [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Empirical p-value (λ_max > λ_qut): {p_val:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(lambda_max_null, bins=50, alpha=0.5, label="λ_max (null)", color="gray")
        plt.hist(lambda_max_bootstrap, bins=50, alpha=0.5, label="λ_max (bootstrap)", color="skyblue")
        plt.axvline(lambda_qut, color="black", linestyle="--", label="λ_qut (95%)")
        plt.axvline(lambda_max_obs, color="red", linestyle="--", label="λ_max (observed)")
        plt.xlabel("λ_max")
        plt.ylabel("Frequency")
        plt.title("λ_max: Bootstrap vs. Null Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "lambda_qut": lambda_qut,
        "lambda_max_obs": lambda_max_obs,
        "ci": (ci_lower, ci_upper),
        "p_value": p_val
    }


# ============================================================
# 6) Beta-map summaries (statistics) + plotting helpers
# ============================================================

def compute_beta_statistics(beta_samples):
    beta_mean = np.mean(beta_samples, axis=0)
    beta_std = np.std(beta_samples, axis=0)
    beta_iqr = np.percentile(beta_samples, 75, axis=0) - np.percentile(beta_samples, 25, axis=0)
    return beta_mean, beta_std, beta_iqr


def compute_beta_summary_stats(beta_samples):
    """
    Compute mean and std dev per cell from bootstrapped beta samples.
    
    Parameters:
        beta_samples (np.ndarray): shape (B, grid_width, grid_height)
        
    Returns:
        beta_mean, beta_std (both np.ndarray): shape (grid_width, grid_height)
    """
    beta_mean = np.mean(beta_samples, axis=0)
    beta_std = np.std(beta_samples, axis=0)
    return beta_mean, beta_std


def plot_beta_spread(beta_mean, beta_std, beta_iqr, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Mean Estimate", "Standard Deviation", "Interquartile Range"]
    matrices = [beta_mean, beta_std, beta_iqr]
    cmaps = ["viridis", "plasma", "inferno"]

    for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
        im = ax.imshow(mat.T, origin="lower", cmap=cmap, extent=[0, 20, 0, 10])
        ax.set_title(f"{title_prefix} {title}")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_beta_mean_std(beta_mean, beta_std, court_image_path=None):
    """
    Visualize mean and standard deviation of beta values across the court grid.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = ["Mean of β (Bootstrapped)", "Standard Deviation of β"]
    matrices = [beta_mean, beta_std]
    cmaps = ["viridis", "plasma"]

    for ax, data, title, cmap in zip(axes, matrices, titles, cmaps):
        if court_image_path:
            from PIL import Image
            court_img = Image.open(court_image_path)
            ax.imshow(court_img, extent=[0, 20, 0, 10], aspect='auto')

        im = ax.imshow(data.T, origin="lower", extent=[0, 20, 0, 10],
                       cmap=cmap, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Court X")
        ax.set_ylabel("Court Y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Pointwise percentile maps at fixed λ (logit scale) + plotting
# ============================================================

def bootstrap_percentile_maps_fixed_lambda(
    boot_datasets,             # list of bootstrapped DataFrames
    lambda_tv,                 # fixed λ (e.g., lambda_qut or lambda_max)
    grid_width, grid_height,   # grid shape
    court_image_path="court.jpg",
    alpha=0.05,
    show_progress=True
):
    """
    Calls tv_logistic_regression_heatmap on each bootstrap with fixed λ.
    Collects *logit* maps and returns percentile summaries on the logit scale.
    """
    B = len(boot_datasets)
    maps = None
    rng = np.random.default_rng(0)

    it = trange(B, desc=f"Bootstraps @ generated λ={lambda_tv:.4g}") if show_progress else range(B)
    for b in it:
        df_b = boot_datasets[b]

        out = tv_logistic_regression_heatmap(
            df_b,
            lambda_tv=lambda_tv,
            grid_width=grid_width,
            grid_height=grid_height,
            court_image_path=court_image_path,
            plot=False  # no plotting in loop
        )

        # Your function may return either logits or (logits, prob_map).
        beta_logit = out[0] if isinstance(out, tuple) else out  # ensure we keep logits
        if maps is None:
            maps = np.empty((B, grid_width, grid_height), dtype=np.float32)
        maps[b] = beta_logit

    # Pointwise percentile summaries on the logit scale
    q_lo, q_hi = alpha/2, 1 - alpha/2
    logit_lo  = np.quantile(maps, q_lo, axis=0)
    logit_med = np.median(maps, axis=0)
    logit_hi  = np.quantile(maps, q_hi, axis=0)

    return logit_med, logit_lo, logit_hi, maps


def plot_three_maps(logit_lo, logit_med, logit_hi, court_image_path, vlim=None):
    # Suggest a symmetric vlim if not provided, using robust percentiles
    if vlim is None:
        v = np.percentile(np.abs(np.stack([logit_lo, logit_med, logit_hi])), 95)
        vlim = (-float(v), float(v))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img = mpimg.imread(court_image_path)

    for ax, data, title in zip(
        axes,
        [logit_lo, logit_med, logit_hi],
        ["Conservative (Lower, logit)", "Realistic (Median, logit)", "Optimistic (Upper, logit)"]
    ):
        ax.imshow(img, extent=[0, data.shape[0], 0, data.shape[1]], aspect='auto')
        im = ax.imshow(
            data.T, origin="lower",
            extent=[0, data.shape[0], 0, data.shape[1]],
            cmap="coolwarm", alpha=0.75, vmin=vlim[0], vmax=vlim[1]
        )
        ax.set_xlabel("x_bin"); ax.set_ylabel("y_bin"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Logit (β̂)")

    plt.tight_layout(); plt.show()


def bin_and_flip(
    df,
    court_length=94,
    court_width=50,
    x_bins=20,
    y_bins=10,
    game_col="game_id",
    period_col="period",
    is_home_col="is_home",
    x_col="x",
    y_col="y",
    flip = "no"
):
    """
    Add x_bin/y_bin and normalize orientation so offensive movement is left-to-right.

    The orientation rule matches the multi-game notebook:
      1. Rotate possessions based on home/away and half.
      2. Rotate any game whose post-rule x-bin mass is still left-heavy.
    """
    out = df.copy()
    mid = (x_bins - 1) / 2

    out["x_bin"] = np.clip((out[x_col] / court_length * x_bins).astype(int), 0, x_bins - 1)
    out["y_bin"] = np.clip((out[y_col] / court_width * y_bins).astype(int), 0, y_bins - 1)

    if flip.lower() == "no":

        def rotate_180(mask):
            out.loc[mask, "x_bin"] = (x_bins - 1) - out.loc[mask, "x_bin"]
            out.loc[mask, "y_bin"] = (y_bins - 1) - out.loc[mask, "y_bin"]

        halftime_flip = (
            ((out[is_home_col] == 1) & (out[period_col] >= 3)) |
            ((out[is_home_col] == 0) & (out[period_col] <= 2))
        )
        rotate_180(halftime_flip)

        game_mean = out.groupby(game_col)["x_bin"].mean()
        games_still_wrong = game_mean[game_mean < mid].index
        rotate_180(out[game_col].isin(games_still_wrong))

    return out


def orient_movement_left_to_right(
    df,
    court_length=94,
    court_width=50,
    x_bins=20,
    game_col="game_id",
    period_col="period",
    is_home_col="is_home",
    x_col="x",
    y_col="y",
    x_out_col="x_oriented",
    y_out_col="y_oriented",
):
    """
    Add oriented x/y columns using the same home/away and per-game rules as bin_and_flip.
    """
    out = df.copy()
    mid = (x_bins - 1) / 2

    out[x_out_col] = out[x_col]
    out[y_out_col] = out[y_col]
    out["_orientation_x_bin"] = np.clip((out[x_col] / court_length * x_bins).astype(int), 0, x_bins - 1)

    def rotate_180(mask):
        out.loc[mask, x_out_col] = court_length - out.loc[mask, x_out_col]
        out.loc[mask, y_out_col] = court_width - out.loc[mask, y_out_col]
        out.loc[mask, "_orientation_x_bin"] = (x_bins - 1) - out.loc[mask, "_orientation_x_bin"]

    halftime_flip = (
        ((out[is_home_col] == 1) & (out[period_col] >= 3)) |
        ((out[is_home_col] == 0) & (out[period_col] <= 2))
    )
    rotate_180(halftime_flip)

    game_mean = out.groupby(game_col)["_orientation_x_bin"].mean()
    games_still_wrong = game_mean[game_mean < mid].index
    rotate_180(out[game_col].isin(games_still_wrong))

    return out.drop(columns="_orientation_x_bin")


def plot_heatmap_bins(
    df_binned,
    img_path,
    ax,
    title="Ball Movement Heatmap",
    x_bins=20,
    y_bins=10,
    cmap="hot",
    alpha=0.6,
    vmin=0,
    vmax=1,
):
    """
    Plot normalized movement density from precomputed x_bin/y_bin columns.
    """
    court_img = mpimg.imread(img_path)
    heatmap = np.zeros((x_bins, y_bins))

    valid = df_binned.dropna(subset=["x_bin", "y_bin"])
    if not valid.empty:
        x_idx = valid["x_bin"].astype(int).clip(0, x_bins - 1)
        y_idx = valid["y_bin"].astype(int).clip(0, y_bins - 1)
        np.add.at(heatmap, (x_idx, y_idx), 1)

    heatmap_norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap

    ax.imshow(court_img, extent=[0, x_bins, 0, y_bins], aspect="auto")
    im = ax.imshow(
        heatmap_norm.T,
        cmap=cmap,
        origin="lower",
        extent=[0, x_bins, 0, y_bins],
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Court Length Bins")
    ax.set_ylabel("Court Width Bins")
    return im


def plot_scoring_movement_heatmaps(
    df,
    img_path="court.jpg",
    scored_col="scored",
    x_bins=20,
    y_bins=10,
    title="Ball Movement Heatmaps by Play Outcome",
    figsize=(16, 6),
):
    """
    Plot side-by-side normalized movement heatmaps for non-scoring and scoring plays.
    """
    binned = df if {"x_bin", "y_bin"}.issubset(df.columns) else bin_and_flip(
        df,
        x_bins=x_bins,
        y_bins=y_bins,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    heatmap_images = []
    plot_specs = [
        (0, "Non-Scoring Plays"),
        (1, "Scoring Plays"),
    ]

    for ax, (scored_value, label) in zip(axes, plot_specs):
        subset = binned[binned[scored_col] == scored_value]
        n_plays = subset["possession_number"].nunique() if "possession_number" in subset else len(subset)
        im = plot_heatmap_bins(
            subset,
            img_path,
            ax,
            title=f"{label} (plays={n_plays})",
            x_bins=x_bins,
            y_bins=y_bins,
            vmin=0,
            vmax=1,
        )
        heatmap_images.append(im)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.94, 0.16, 0.015, 0.68])
    fig.colorbar(heatmap_images[0], cax=cbar_ax, label="Normalized Ball Movement Density")
    plt.show()
    return fig, axes


def plot_scoring_movement_heatmaps_by_game(
    df,
    img_path="court.jpg",
    game_col="game_id",
    scored_col="scored",
    x_bins=20,
    y_bins=10,
    title="Ball Movement Heatmaps by Game and Play Outcome",
    row_height=3.4,
    figsize=None,
):
    """
    Plot normalized movement heatmaps for non-scoring and scoring plays within each game.
    """
    binned = df if {"x_bin", "y_bin"}.issubset(df.columns) else bin_and_flip(
        df,
        x_bins=x_bins,
        y_bins=y_bins,
        game_col=game_col,
    )

    games = sorted(binned[game_col].dropna().unique())
    if not games:
        raise ValueError(f"No games found in column '{game_col}'.")

    if figsize is None:
        figsize = (16, max(4, row_height * len(games)))

    fig, axes = plt.subplots(len(games), 2, figsize=figsize, squeeze=False)
    heatmap_images = []
    plot_specs = [
        (0, "Non-Scoring"),
        (1, "Scoring"),
    ]

    for row, game in enumerate(games):
        game_df = binned[binned[game_col] == game]
        for col, (scored_value, label) in enumerate(plot_specs):
            ax = axes[row, col]
            subset = game_df[game_df[scored_col] == scored_value]
            n_plays = subset["possession_number"].nunique() if "possession_number" in subset else len(subset)
            im = plot_heatmap_bins(
                subset,
                img_path,
                ax,
                title=f"{game} | {label} (plays={n_plays})",
                x_bins=x_bins,
                y_bins=y_bins,
                vmin=0,
                vmax=1,
            )
            heatmap_images.append(im)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 0.92, 0.97])
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    fig.colorbar(heatmap_images[0], cax=cbar_ax, label="Normalized Ball Movement Density")
    plt.show()
    return fig, axes


def plot_raw_movements_by_game(
    df,
    img_path="court.jpg",
    game_col="game_id",
    possession_col="possession_number",
    scored_col="scored",
    time_col="time",
    x_col="x",
    y_col="y",
    period_col="period",
    is_home_col="is_home",
    normalize_orientation=True,
    court_length=94,
    court_width=50,
    title="Raw Ball Movement by Game and Play Outcome",
    row_height=3.4,
    figsize=None,
    line_alpha=0.55,
    line_width=1.2,
    marker_size=18,
):
    """
    Plot raw possession paths for non-scoring and scoring plays within each game.
    """
    plot_df = orient_movement_left_to_right(
        df,
        court_length=court_length,
        court_width=court_width,
        game_col=game_col,
        period_col=period_col,
        is_home_col=is_home_col,
        x_col=x_col,
        y_col=y_col,
    ) if normalize_orientation else df.copy()

    plot_x = "x_oriented" if normalize_orientation else x_col
    plot_y = "y_oriented" if normalize_orientation else y_col

    games = sorted(plot_df[game_col].dropna().unique())
    if not games:
        raise ValueError(f"No games found in column '{game_col}'.")

    if figsize is None:
        figsize = (18, max(4, row_height * len(games)))

    court_img = mpimg.imread(img_path)
    fig, axes = plt.subplots(len(games), 2, figsize=figsize, squeeze=False)
    plot_specs = [
        (0, "Non-Scoring"),
        (1, "Scoring"),
    ]

    for row, game in enumerate(games):
        game_df = plot_df[plot_df[game_col] == game]
        for col, (scored_value, label) in enumerate(plot_specs):
            ax = axes[row, col]
            subset = game_df[game_df[scored_col] == scored_value]
            possession_ids = subset[possession_col].dropna().unique()
            cmap = plt.get_cmap("tab20", max(len(possession_ids), 1))

            ax.imshow(court_img, extent=[0, court_length, 0, court_width], zorder=0)
            ax.set_xlim(0, court_length)
            ax.set_ylim(0, court_width)
            ax.set_title(f"{game} | {label} (plays={len(possession_ids)})")
            ax.set_xlabel("Court X")
            ax.set_ylabel("Court Y")

            for i, possession_id in enumerate(possession_ids):
                group = subset[subset[possession_col] == possession_id]
                if time_col in group:
                    group = group.sort_values(by=time_col, ascending=False)
                if group.empty:
                    continue

                color = cmap(i)
                ax.plot(
                    group[plot_x],
                    group[plot_y],
                    color=color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=1,
                )
                ax.scatter(
                    group.iloc[0][plot_x],
                    group.iloc[0][plot_y],
                    marker="s",
                    color="blue",
                    s=marker_size,
                    zorder=2,
                )
                ax.scatter(
                    group.iloc[-1][plot_x],
                    group.iloc[-1][plot_y],
                    marker="X",
                    color="red",
                    s=marker_size,
                    zorder=2,
                )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()
    return fig, axes


def plot_raw_movements_by_game_period(
    df,
    img_path="court.jpg",
    game_col="game_id",
    period_col="period",
    possession_col="possession_number",
    scored_col="scored",
    time_col="time",
    x_col="x",
    y_col="y",
    is_home_col="is_home",
    normalize_orientation=True,
    court_length=94,
    court_width=50,
    title="Raw Ball Movement by Game, Period, and Play Outcome",
    row_height=3.0,
    figsize=None,
    line_alpha=0.55,
    line_width=1.2,
    marker_size=18,
):
    """
    Plot raw possession paths by game and period, split into non-scoring and scoring plays.
    """
    plot_df = orient_movement_left_to_right(
        df,
        court_length=court_length,
        court_width=court_width,
        game_col=game_col,
        period_col=period_col,
        is_home_col=is_home_col,
        x_col=x_col,
        y_col=y_col,
    ) if normalize_orientation else df.copy()

    plot_x = "x_oriented" if normalize_orientation else x_col
    plot_y = "y_oriented" if normalize_orientation else y_col

    game_periods = (
        plot_df[[game_col, period_col]]
        .dropna()
        .drop_duplicates()
        .sort_values([game_col, period_col])
        .itertuples(index=False, name=None)
    )
    game_periods = list(game_periods)
    if not game_periods:
        raise ValueError(f"No game/period combinations found in '{game_col}' and '{period_col}'.")

    if figsize is None:
        figsize = (18, max(4, row_height * len(game_periods)))

    court_img = mpimg.imread(img_path)
    fig, axes = plt.subplots(len(game_periods), 2, figsize=figsize, squeeze=False)
    plot_specs = [
        (0, "Non-Scoring"),
        (1, "Scoring"),
    ]

    for row, (game, period) in enumerate(game_periods):
        period_df = plot_df[(plot_df[game_col] == game) & (plot_df[period_col] == period)]
        for col, (scored_value, label) in enumerate(plot_specs):
            ax = axes[row, col]
            subset = period_df[period_df[scored_col] == scored_value]
            possession_ids = subset[possession_col].dropna().unique()
            cmap = plt.get_cmap("tab20", max(len(possession_ids), 1))

            ax.imshow(court_img, extent=[0, court_length, 0, court_width], zorder=0)
            ax.set_xlim(0, court_length)
            ax.set_ylim(0, court_width)
            ax.set_title(f"{game} | Period {period} | {label} (plays={len(possession_ids)})")
            ax.set_xlabel("Court X")
            ax.set_ylabel("Court Y")

            for i, possession_id in enumerate(possession_ids):
                group = subset[subset[possession_col] == possession_id]
                if time_col in group:
                    group = group.sort_values(by=time_col, ascending=False)
                if group.empty:
                    continue

                color = cmap(i)
                ax.plot(
                    group[plot_x],
                    group[plot_y],
                    color=color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=1,
                )
                ax.scatter(
                    group.iloc[0][plot_x],
                    group.iloc[0][plot_y],
                    marker="s",
                    color="blue",
                    s=marker_size,
                    zorder=2,
                )
                ax.scatter(
                    group.iloc[-1][plot_x],
                    group.iloc[-1][plot_y],
                    marker="X",
                    color="red",
                    s=marker_size,
                    zorder=2,
                )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()
    return fig, axes


# ============================================================
# 8) End-to-end per-period analysis pipeline
# ============================================================

def analyze_tv_logistic_single_game(
    df_game,
    game_id=None,
    game_col="game_id",
    n_bootstraps=250,
    n_possessions=1000,
    samples_per_possession=100,
    min_samples_required=100,
    grid_width=20,
    grid_height=10,
    mc_null=250,
    alpha=0.05,
    court_image_path="court.jpg",
    show_progress=True,
    plot_beta_summary=False,
    plot_pointwise_maps=False,
    plot_lambda_distributions=True,
    seed=42
):
    """
    Run the bootstrap/QUT workflow for one game.

    The ordering intentionally matches the bootstrap-first workflow:
      1. Build possession-level bootstrap datasets from the selected game.
      2. Compute QUT and the displayed lambda_max on boot_datasets[0].
      3. Compute the remaining bootstrap lambda_max values on boot_datasets[1:].
      4. Summarize the null and bootstrap lambda_max distributions.

    Parameters:
    - df_game: binned/oriented tracking DataFrame with x_bin, y_bin, possession_number, scored.
    - game_id: optional value used to filter df_game[game_col] before analysis.
    - plot_beta_summary / plot_pointwise_maps: enable the heavier beta-map portion.

    Returns:
    - dict containing lambda statistics, raw distributions, bootstraps, and optional pointwise maps.
    """
    if game_id is not None:
        if game_col not in df_game.columns:
            raise ValueError(f"game_id was provided, but column '{game_col}' is not in df_game.")
        df_analysis = df_game[df_game[game_col] == game_id].copy()
    else:
        df_analysis = df_game.copy()

    if df_analysis.empty:
        label = f" for game {game_id}" if game_id is not None else ""
        raise ValueError(f"No rows available{label}.")

    required_cols = {"possession_number", "x_bin", "y_bin", "scored"}
    missing_cols = required_cols - set(df_analysis.columns)
    if missing_cols:
        raise ValueError(f"df_game is missing required columns: {sorted(missing_cols)}")

    label = f" game {game_id}" if game_id is not None else ""
    print(f"\n=== Processing single{label} ===")

    boot_datasets = generate_bootstrapped_datasets(
        df_analysis,
        n_bootstraps=n_bootstraps,
        n_possessions_per_bootstrap=n_possessions,
        samples_per_possession=samples_per_possession,
        min_samples_required=min_samples_required,
        seed=seed
    )

    df_boot = boot_datasets[0]
    X, y, D_TV = grid(df_boot)
    lambda_qut_val, lambda_max_obs, lambdas_null = lambda_qut(X, y, D_TV, MC=mc_null)

    lambda_max_bootstrap_rest = compute_lambda_max_from_bootstraps(
        boot_datasets[1:],
        D_TV,
        solve_lambda_func=solve_infinity_norm,
        project_grad_h=True
    )
    lambda_max_bootstrap = np.concatenate([
        np.asarray([lambda_max_obs], dtype=float),
        np.asarray(lambda_max_bootstrap_rest, dtype=float)
    ])

    stats = summarize_lambda_max_analysis(
        lambda_max_obs=float(lambda_max_obs),
        lambda_max_null=np.asarray(lambdas_null, dtype=float),
        lambda_max_bootstrap=np.asarray(lambda_max_bootstrap, dtype=float),
        plot=plot_lambda_distributions
    )

    result = {
        "game_id": game_id,
        "boot_datasets": boot_datasets,
        "lambda_qut": float(lambda_qut_val),
        "lambda_max_obs": float(stats["lambda_max_obs"]),
        "lambda_max_ci": stats["ci"],
        "p_value": stats["p_value"],
        "lambda_null": np.asarray(lambdas_null, dtype=float),
        "lambda_max_bootstrap": np.asarray(lambda_max_bootstrap, dtype=float),
        "summary": stats,
    }

    if plot_beta_summary or plot_pointwise_maps:
        logit_med, logit_lo, logit_hi, maps_all = bootstrap_percentile_maps_fixed_lambda(
            boot_datasets=boot_datasets,
            lambda_tv=lambda_qut_val,
            grid_width=grid_width,
            grid_height=grid_height,
            court_image_path=court_image_path,
            alpha=alpha,
            show_progress=show_progress
        )

        beta_mean, beta_std = compute_beta_summary_stats(maps_all)
        if plot_beta_summary:
            plot_beta_mean_std(beta_mean, beta_std, court_image_path=court_image_path)

        if plot_pointwise_maps:
            print(f"Plotting pointwise percentile maps at lambda = {lambda_qut_val:.4f}")
            plot_three_maps(logit_lo, logit_med, logit_hi, court_image_path=court_image_path)

        result.update({
            "beta_samples": maps_all,
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "pointwise": {
                "lambda": float(lambda_qut_val),
                "alpha": float(alpha),
                "logit_lo": logit_lo,
                "logit_med": logit_med,
                "logit_hi": logit_hi,
                "maps": maps_all,
            }
        })

    return result


def analyze_tv_logistic_per_period(
    df_original,
    n_bootstraps=100,
    n_possessions=500,
    samples_per_possession=100,
    min_samples_required=100,
    grid_width=20,
    grid_height=10,
    # null + plotting controls
    mc_null=500,                      # number of Monte Carlo null simulations per period
    alpha=0.05,                       # confidence level for pointwise maps
    court_image_path="court.jpg",
    show_progress=True,
    plot_beta_summary=True,
    plot_pointwise_maps=True,
    plot_lambda_distributions=True,
    seed_base=42
):
    """
    For each period:
      • Builds bootstrap datasets
      • Computes λ_qut for that period via Monte Carlo null simulations
      • Computes β summaries (mean/std) at λ = λ_qut(period)
      • Computes λ statistics (λ_max_obs, bootstrap CI, p-value) using summarize_lambda_max_analysis()
      • Computes pointwise percentile maps (logit) at λ = λ_qut(period)

    Notes:
      - λ_qut is computed separately for each period using the same D_TV structure.
      - The p-value returned by summarize_lambda_max_analysis is computed as
        mean(lambda_max_null >= lambda_max_obs).
    """
    rng = np.random.default_rng(seed_base)
    period_results = {}
    periods = sorted(df_original['period'].unique())

    for period in periods:
        print(f"\n=== Processing Period {period} ===")

        df_period = df_original[df_original['period'] == period]

        # ---------- Step 0: Bootstrap datasets ----------
        boot_datasets = generate_bootstrapped_datasets(
            df_period,
            n_bootstraps=n_bootstraps,
            n_possessions_per_bootstrap=n_possessions,
            samples_per_possession=samples_per_possession,
            min_samples_required=min_samples_required,
            seed=seed_base + period
        )

        # ---------- Step 1: Compute λ_qut for this period ----------
        df_boot = boot_datasets[0]
        X, y, D_TV = grid(df_boot)

        lambda_qut_val, lambda_max_obs, lambdas_null = lambda_qut(X, y, D_TV, MC=mc_null)
        print(f"λ_qut (period {period}): {lambda_qut_val:.4f}")

        # ---------- Step 2: Bootstrap λ_max and summarize ----------
        lambda_max_bootstrap_rest = compute_lambda_max_from_bootstraps(
            boot_datasets[1:],
            D_TV,
            solve_lambda_func=solve_infinity_norm
        )
        lambda_max_bootstrap = np.concatenate([
            np.asarray([lambda_max_obs], dtype=float),
            np.asarray(lambda_max_bootstrap_rest, dtype=float)
        ])

        stats = summarize_lambda_max_analysis(
            lambda_max_obs=float(lambda_max_obs),
            lambda_max_null=np.asarray(lambdas_null, dtype=float),
            lambda_max_bootstrap=np.asarray(lambda_max_bootstrap, dtype=float),
            plot=plot_lambda_distributions
        )

        # ---------- Step 3: Pointwise percentile maps using λ_qut ----------
        logit_med, logit_lo, logit_hi, maps_all = bootstrap_percentile_maps_fixed_lambda(
            boot_datasets=boot_datasets,
            lambda_tv=lambda_qut_val,
            grid_width=grid_width,
            grid_height=grid_height,
            court_image_path=court_image_path,
            alpha=alpha,
            show_progress=show_progress
        )

        # ---------- Step 4: β summaries using the fitted pointwise maps ----------
        beta_samples = maps_all
        beta_mean, beta_std = compute_beta_summary_stats(beta_samples)
        if plot_beta_summary:
            plot_beta_mean_std(beta_mean, beta_std, court_image_path=court_image_path)

        if plot_pointwise_maps:
            print(f"Plotting pointwise percentile maps at λ = {lambda_qut_val:.4f}")
            plot_three_maps(logit_lo, logit_med, logit_hi, court_image_path=court_image_path)

        # ---------- Step 5: Pack results ----------
        period_results[period] = {
            # β summaries at λ_qut
            "beta_samples": beta_samples,
            "beta_mean": beta_mean,
            "beta_std": beta_std,

            # λ statistics
            "lambda_qut": lambda_qut_val,
            "lambda_max_obs": stats["lambda_max_obs"],
            "lambda_max_ci": stats["ci"],
            "p_value": stats["p_value"],

            # raw distributions
            "lambda_null": np.asarray(lambdas_null, dtype=float),
            "lambda_max_bootstrap": np.asarray(lambda_max_bootstrap, dtype=float),

            # Pointwise percentile maps
            "pointwise": {
                "lambda": float(lambda_qut_val),
                "alpha": float(alpha),
                "logit_lo": logit_lo,
                "logit_med": logit_med,
                "logit_hi": logit_hi,
                "maps": maps_all,   # shape: (B, grid_width, grid_height)
            }
        }

    return period_results
