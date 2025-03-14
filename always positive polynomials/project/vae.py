import os
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sympy import symbols, solveset, S, lambdify

project_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(project_dir, "polynomials.csv")

############################################################################
# GLOBALS
############################################################################
x = symbols('x')  # for optional symbolic check if needed
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(suppress=True, precision=4)

############################################################################
# 1) Symbolic / Domain Checks
############################################################################
def is_always_positive(coeffs):
    """
    Symbolic check: tries to verify if P(x) > 0 for all x in R.
    Not 100% foolproof for high degrees, but often works as an approximation.
    """
    coeffs = np.trim_zeros(coeffs, trim='f')
    if len(coeffs) == 0:
        return False
    deg = len(coeffs) - 1
    if coeffs[0] <= 0:
        return False
    if deg % 2 != 0:
        return False

    poly_expr = sum(c * x**i for i, c in enumerate(coeffs[::-1]))
    derivative = poly_expr.diff(x)
    critical_points = solveset(derivative, x, domain=S.Reals)
    poly_func = lambdify(x, poly_expr)

    if critical_points not in [S.EmptySet, S.Reals]:
        for cp in critical_points:
            try:
                val = poly_func(float(cp))
                if val <= 0:
                    return False
            except:
                return False
    return True


def poly_eval(coeffs, x_vals):
    """
    coeffs: shape (batch_size, poly_degree+1)
    x_vals: shape (num_x_samples,)
    returns pvals: shape (batch_size, num_x_samples)
    """
    batch_size, deg_plus_1 = coeffs.shape
    degree = deg_plus_1 - 1
    pvals = torch.zeros(batch_size, x_vals.shape[0], device=coeffs.device)
    for i in range(degree + 1):
        pvals += coeffs[:, i].unsqueeze(1) * (x_vals ** i)
    return pvals


def is_positive_in_domain(coeffs_batch, n_samples=10, x_min=-10.0, x_max=10.0):
    """
    For each polynomial in coeffs_batch, check if it is > 0 for all sampled points in [x_min, x_max].
    Returns a boolean mask of shape (batch_size,).
    """
    x_vals = torch.linspace(x_min, x_max, n_samples, device=coeffs_batch.device)
    pvals = poly_eval(coeffs_batch, x_vals)  
    return (pvals > 0).all(dim=1)


def positivity_penalty(coeffs, n_samples=10, x_min=-10.0, x_max=10.0):
    """
    penalty = sum of negative parts across x-samples, averaged over batch.
    """
    x_vals = torch.linspace(x_min, x_max, n_samples, device=coeffs.device)
    pvals = poly_eval(coeffs, x_vals)
    penalty = torch.relu(-pvals).sum(dim=1)  # sum across x-samples
    return penalty.mean()


############################################################################
# 2) Weighted Data Approach
############################################################################
def compute_sample_weight(coeff_row):
    """
    Given a row of polynomial coefficients (as numpy array),
    we check if it's already positive in [-10,10].
    If it is, we assign a higher weight, else 1.0.
    """
    # Quick domain check (in CPU mode).
    # We'll do a small check with 10 points between -10 and 10.
    # This is not symbolic, but enough to see if it's mostly positive.
    n_samples = 10
    x_vals = np.linspace(-10, 10, n_samples)
    deg = len(coeff_row) - 1

    # Evaluate polynomial for these x_vals
    pvals = np.zeros(n_samples, dtype=np.float32)
    for i in range(deg + 1):
        pvals += coeff_row[i] * (x_vals ** i)

    # If all pvals>0 => return 2.0 else 1.0
    if np.all(pvals > 0):
        return 2.0  # you can pick 2.0, 3.0, etc. for stronger weighting
    else:
        return 1.0


############################################################################
# 3) CSV Utility
############################################################################
def detect_max_degree_in_csv(df):
    candidate_cols = [c for c in df.columns if c.startswith('a')]
    if not candidate_cols:
        raise ValueError("No coefficient columns found in DataFrame.")
    degrees_found = []
    for col in candidate_cols:
        num_part = col[1:]
        degrees_found.append(int(num_part))
    return max(degrees_found)


############################################################################
# 4) VAE Model
############################################################################
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.log_var_layer = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


############################################################################
# 5) Custom VAE Loss
############################################################################
def custom_vae_loss(recon, original, mu, log_var, penalty_weight=0.1):
    mse = nn.MSELoss()(recon, original)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    base_loss = mse + kl_div / original.size(0)

    pos_pen = positivity_penalty(recon, n_samples=10, x_min=-10, x_max=10)
    return base_loss + penalty_weight * pos_pen, mse.item(), pos_pen.item()


############################################################################
# 6) Train with WeightedRandomSampler, Higher Penalty, & Scheduler
############################################################################
from torch.optim.lr_scheduler import StepLR

def train_vae_for_degree(df, deg, epochs=5, penalty_weight=1.0):
    """
    1) Filter df => only rows with 'degree' = deg
    2) Re-weight the data: polynomials that are already positive in [-10,10] get bigger weight.
    3) Use WeightedRandomSampler for the training set.
    4) Setup a StepLR scheduler to reduce LR every 'step_size' epochs.
    """
    max_degree_in_csv = detect_max_degree_in_csv(df)
    coeff_cols = [f"a{i}" for i in range(max_degree_in_csv, -1, -1)]

    df_deg = df[df["degree"] == deg].copy()
    print(f"Training VAE on degree={deg}, data size={len(df_deg)}")

    # fill missing columns
    for col in coeff_cols:
        if col not in df_deg.columns:
            df_deg[col] = 0.0

    feats = df_deg[coeff_cols].values  # shape: [N, max_degree_in_csv+1]

    # zero out columns above deg
    total_cols = feats.shape[1]
    keep = deg + 1
    slice_end = total_cols - keep
    if slice_end > 0:
        for row in feats:
            row[:slice_end] = 0.0

    # A) compute weights per sample
    #    e.g. polynomials already positive => weight=2.0, else=1.0
    sample_weights = []
    for row in feats:
        w = compute_sample_weight(row)
        sample_weights.append(w)
    sample_weights = np.array(sample_weights, dtype=np.float32)

    # B) split
    train_data, val_data, train_w, val_w = train_test_split(
        feats, sample_weights, test_size=0.2, random_state=42
    )

    class PolyDataset(Dataset):
        def __init__(self, arr):
            self.data = torch.tensor(arr, dtype=torch.float32)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = PolyDataset(train_data)
    val_ds = PolyDataset(val_data)

    # WeightedRandomSampler wants weights for each sample
    train_sampler = WeightedRandomSampler(weights=train_w, num_samples=len(train_w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    input_dim = len(coeff_cols)
    model = VAE(input_dim=input_dim, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # C) learning rate scheduler: reduce LR by factor of 0.1 every 50 epochs
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, mse_val, pos_val = custom_vae_loss(
                recon, batch, mu, log_var, penalty_weight
            )
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # domain-based always-positive check
            with torch.no_grad():
                pos_mask = is_positive_in_domain(recon, n_samples=10, x_min=-10, x_max=10)
                batch_acc = pos_mask.float().mean().item()
            total_train_acc += batch_acc

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, log_var = model(batch)
                loss, mse_val, pos_val = custom_vae_loss(
                    recon, batch, mu, log_var, penalty_weight
                )
                total_val_loss += loss.item()

                pos_mask = is_positive_in_domain(recon, n_samples=10, x_min=-10, x_max=10)
                batch_acc = pos_mask.float().mean().item()
                total_val_acc += batch_acc

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)

        # Step the scheduler after each epoch
        scheduler.step()

        if(epoch%25== 0):
            print(f"[Degree={deg}, Epoch={epoch+1}/{epochs}] "
                f"TrainLoss={avg_train_loss:.4f}, TrainAcc={avg_train_acc:.4f} | "
                f"ValLoss={avg_val_loss:.4f}, ValAcc={avg_val_acc:.4f} "
                f"(LR={optimizer.param_groups[0]['lr']:.5f})")

    return model


############################################################################
# 7) The Fallback Logic & Generation Remain Similar
############################################################################
def find_fallback_degree(df, requested_degree, min_samples=25):
    """
    Finds the nearest lower even degree with at least `min_samples` in the dataset.
    If none are found, defaults to degree 2.
    
    Parameters:
    - df: DataFrame containing the polynomial dataset.
    - requested_degree: The degree requested by the user.
    - min_samples: Minimum required samples for the degree.

    Returns:
    - deg_candidate: Fallback degree with enough samples.
    """
    deg_candidate = requested_degree
    while deg_candidate >= 2:
        count_deg = df[df["degree"] == deg_candidate].shape[0]
        if count_deg >= min_samples:
            return deg_candidate
        deg_candidate -= 2
    return 2  # Default fallback

def find_max_valid_degree(df, min_samples=25):
    """
    Finds the maximum even degree in the dataset that has at least `min_samples`.

    Parameters:
    - df: DataFrame containing the polynomial dataset.
    - min_samples: Minimum required samples for a degree to be considered valid.

    Returns:
    - max_valid_degree: The highest even degree with enough samples, or 2 if none are found.
    """
    # Get all unique even degrees in descending order
    even_degrees = sorted(df["degree"].unique(), reverse=True)
    even_degrees = [deg for deg in even_degrees if deg % 2 == 0]

    # Find the highest degree with enough samples
    for deg in even_degrees:
        if df[df["degree"] == deg].shape[0] >= min_samples:
            return deg

    return 2  # Default fallback if no valid degree is found


def find_fallback_degree(df, requested_degree, min_samples=25):
    """
    Finds the nearest lower even degree with at least `min_samples` in the dataset.
    If none are found, defaults to degree 2.
    
    Parameters:
    - df: DataFrame containing the polynomial dataset.
    - requested_degree: The degree requested by the user.
    - min_samples: Minimum required samples for the degree.

    Returns:
    - deg_candidate: Fallback degree with enough samples.
    """
    deg_candidate = requested_degree
    while deg_candidate >= 2:
        count_deg = df[df["degree"] == deg_candidate].shape[0]
        if count_deg >= min_samples:
            return deg_candidate
        deg_candidate -= 2
    return 2  # Default fallback

def generate_custom_degree_polynomial(model, base_degree, final_degree, max_attempts=2000, csv_path= CSV_PATH):
    base_poly = None
    tries = 0
    while tries < max_attempts:
        tries += 1
        latent = torch.randn(1, 8, device=device)
        decoded_full = model.decode(latent).detach().cpu().numpy().flatten()
        keep = base_degree + 1
        base_poly = decoded_full[-keep:]
        if (is_always_positive(base_poly)== True):
            break
        else:
            print("Generated polynomial is not positive, trying again...") 

    if base_poly is None:
        raise RuntimeError(f"Could not generate base polynomial for degree={base_degree} in {max_attempts} attempts.")

    if final_degree == base_degree:
        return base_poly
      
    # Random polynomial saved to CSV without positivity check
    random_poly = np.random.uniform(-1, 1, final_degree + 1).astype(np.float32)
    append_poly_to_csv(random_poly, final_degree, csv_path)
    print(f"Random polynomial of degree {final_degree} appended to CSV.")

    top_count = final_degree - base_degree
    if top_count < 0:
        raise ValueError("final_degree < base_degree?")

    poly_out = np.zeros(final_degree + 1, dtype=np.float32)

    tries = 0
    while tries < max_attempts:
        tries += 1
        new_top = np.random.uniform(-1, 1, size=top_count)
        poly_out[0:top_count] = new_top
        poly_out[top_count:] = base_poly
        if is_always_positive(poly_out):
            return poly_out

    raise RuntimeError(f"Could not create a {final_degree}-degree polynomial after {max_attempts} tries.")


def generate_poly_with_fallback(df, requested_degree, epochs=5, penalty_weight=1.0, csv_file= CSV_PATH):
    if requested_degree % 2 != 0:
        raise ValueError("Requested an odd degree => not guaranteed always positive.")
    
    deg_candidate = find_fallback_degree(df, requested_degree)


    if deg_candidate < requested_degree:
        print(f"Using fallback degree={deg_candidate}, because {requested_degree} had <25 rows.")

    model = train_vae_for_degree(df, deg_candidate, epochs=epochs, penalty_weight=penalty_weight)
    final_poly = generate_custom_degree_polynomial(model, base_degree=deg_candidate, final_degree=requested_degree, csv_path=csv_file)
    return final_poly


def append_poly_to_csv(poly_coeffs, final_degree, csv_path):
    needed_count = final_degree + 1
    poly_sliced = poly_coeffs[-needed_count:]
    needed_cols = [f"a{i}" for i in range(final_degree, -1, -1)] + ["degree", "label"]

    # Check if the polynomial is always positive
    label = 1 if is_always_positive(poly_sliced) else 0

    row_data = list(poly_sliced) + [final_degree, label]

    df_new = pd.DataFrame([row_data], columns=needed_cols)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
    else:
        df_existing = pd.DataFrame()

    existing_cols = list(df_existing.columns)
    missing_cols = [c for c in needed_cols if c not in existing_cols]
    for col in reversed(missing_cols):
        if col not in existing_cols:
            existing_cols.insert(0, col)

    df_existing = df_existing.reindex(columns=existing_cols)
    for c in df_existing.columns:
        if c not in df_new.columns:
            df_new[c] = np.nan

    df_new = df_new.reindex(columns=df_existing.columns)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(csv_path, index=False)
    print(f"Appended polynomial of degree={final_degree} with label={label} to CSV.")


def main():
    csv_file = CSV_PATH
    if not os.path.exists(csv_file):
        print("CSV file does not exist.")
        return

    df_all = pd.read_csv(csv_file)
    df_all["degree"] = pd.to_numeric(df_all["degree"], errors="coerce")
    df_all.dropna(subset=["degree"], inplace=True)
    df_all["degree"] = df_all["degree"].astype(int)

    while True:
        user_in = input("Enter an even degree polynomial to generate (or 'quit'): ").strip().lower()
        if user_in == "quit":
            break
        try:
            requested_degree = int(user_in)
        except:
            print("Invalid input.")
            continue

        if requested_degree < 2 or requested_degree % 2 != 0:
            print("Please provide an even degree >= 2.")
            continue

        try:
            # Use bigger penalty_weight=1.0, train more epochs to see effect
            poly = generate_poly_with_fallback(df_all, requested_degree, epochs=200, penalty_weight=1.0)
            print(f"Generated polynomial for degree={requested_degree}:\n{poly}")
            print(f"Symbolic positivity check: {is_always_positive(poly)}")

            append_poly_to_csv(poly, requested_degree, csv_file)
        except RuntimeError as e:
            print("Error generating polynomial:", e)


if __name__ == "__main__":
    main()