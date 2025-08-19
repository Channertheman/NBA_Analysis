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

def solve_infinity_norm(y, X, D, project_grad_h=False):
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
        raise ValueError(f"Optimization failed. Status: {prob.status}")

    return w.value, prob.value

def lambda_qut(X, y, D_TV, MC=100):
    X0 = X @ np.ones((X.shape[1], 1))
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs')
    clf.fit(X0, y)
    x0 = clf.coef_.flatten()
    mu0 = X0 @ x0
    p0 = 1 / (1 + np.exp(-mu0))
    p0 = np.clip(p0, 1e-6, 1 - 1e-6)

    Lambdas = []
    attempts = 0

    print("Simulating null responses...")
    while len(Lambdas) < MC and attempts < MC * 20:
        attempts += 1
        y_sim = binom.rvs(1, p0)

        if np.all(y_sim == 0) or np.all(y_sim == 1):
            print(f"Skipping degenerate simulation (all {int(y_sim[0])})")
            continue

        try:
            _, lambda_val = solve_infinity_norm(y_sim, X, D_TV, project_grad_h=True)
            if np.isfinite(lambda_val):
                Lambdas.append(lambda_val)
                if len(Lambdas) % 100 == 0:
                    print(f"Accepted {len(Lambdas)} / {MC} simulations")
            else:
                print("Skipping non-finite lambda")
        except Exception as e:
            print(f"Skipping simulation due to error: {e}")

    if len(Lambdas) == 0:
        raise RuntimeError("All simulations failed. Check your data or D matrix.")

    lambda_qut_val = np.quantile(Lambdas, 0.95)

    try:
        _, lambda_max = solve_infinity_norm(y, X, D_TV, project_grad_h=True)
    except Exception as e:
        raise RuntimeError(f"Optimization on real data failed: {e}")

    print(f"λ_qut (95th percentile null): {lambda_qut_val:.4f}")
    print(f"λ_max (data): {lambda_max:.4f}")
    return lambda_qut_val, lambda_max, Lambdas

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

def compute_beta_statistics(beta_samples):
    beta_mean = np.mean(beta_samples, axis=0)
    beta_std = np.std(beta_samples, axis=0)
    beta_iqr = np.percentile(beta_samples, 75, axis=0) - np.percentile(beta_samples, 25, axis=0)
    return beta_mean, beta_std, beta_iqr

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

