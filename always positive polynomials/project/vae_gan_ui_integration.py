import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

project_dir = os.path.dirname(os.path.abspath(__file__))

# === Dynamic Import of VAE Functions === #
VAE_FILE_PATH = os.path.join(project_dir, "vae.py")

vae_spec = importlib.util.spec_from_file_location("vae_module", VAE_FILE_PATH)
vae_module = importlib.util.module_from_spec(vae_spec)
sys.modules["vae_module"] = vae_module
vae_spec.loader.exec_module(vae_module)

from vae_module import generate_poly_with_fallback, is_always_positive, append_poly_to_csv, find_max_valid_degree

# === Dynamic Import of GAN Functions === #
GAN_FILE_PATH = os.path.join(project_dir, "gan.py")

gan_spec = importlib.util.spec_from_file_location("gan_module", GAN_FILE_PATH)
gan_module = importlib.util.module_from_spec(gan_spec)
sys.modules["gan_module"] = gan_module
gan_spec.loader.exec_module(gan_module)

from gan_module import generate_poly_with_gan

# === Constants === #
CSV_FILE = os.path.join(project_dir, "polynomials.csv")

# === Streamlit UI === #
st.title("📐 Polynomial Generator with VAE / GAN")

# === VAE and GAN Descriptions === #
st.markdown("""
### **Variational Autoencoder (VAE)**
- VAE is a probabilistic model that represents and reconstructs data in a low-dimensional latent space. In polynomial generation, it provides controllable and stable results. Our accuracy of the model is more than %90.

### **Generative Adversarial Network (GAN)**
- A GAN model consists of two neural networks: a generator and a discriminator. The generator produces “fake” polynomials, while the discriminator attempts to distinguish whether those polynomials are real or fake. Our accuracy of the model is nearly %80.
""")

# Initialize session state variables
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "generation_done" not in st.session_state:
    st.session_state.generation_done = False
if "gan_failures" not in st.session_state:
    st.session_state.gan_failures = {}
if "new_poly_clicked" not in st.session_state:
    st.session_state.new_poly_clicked = False

# === Load CSV Data === #
df_all = pd.read_csv(CSV_FILE)
df_all["degree"] = pd.to_numeric(df_all["degree"], errors="coerce")
df_all.dropna(subset=["degree"], inplace=True)
df_all["degree"] = df_all["degree"].astype(int)

# === Find Max Valid Degree + 4 === #
max_valid_degree = find_max_valid_degree(df_all)
max_allowed_degree = max_valid_degree + 4

# === Display Maximum Allowed Degree === #
st.info(f"⚠️ Maximum allowed degree is **{max_allowed_degree}**. Please enter an even degree ≤ **{max_allowed_degree}**.")

# === Sidebar Model Selection === #
model_choice = st.sidebar.radio("Select Model for Generation:", ["VAE", "GAN"])

# === Degree Input === #
requested_degree = st.number_input("Enter an even degree for polynomial generation:", min_value=2, step=2)

# Error Flag to Disable Button
input_error = False
if requested_degree % 2 != 0:
    st.error("❌ Odd-degree polynomials cannot always be positive. Please enter an even degree.")
    input_error = True
elif requested_degree > max_allowed_degree:
    st.error(f"❌ Maximum allowed degree is {max_allowed_degree}. Please enter a degree ≤ {max_allowed_degree}.")
    input_error = True

# === Button click handler === #
def start_generation():
    st.session_state.is_generating = True
    st.session_state.generation_done = False

generate_button = st.button("Generate Polynomial", on_click=start_generation, disabled=st.session_state.is_generating or input_error)

# Refresh Button
if st.button("🔄 New Polynomial"):
    st.session_state.is_generating = False
    st.session_state.generation_done = False
    st.session_state.gan_failures = {}
    st.session_state.new_poly_clicked = True
    st.rerun()

# === Format Polynomial to LaTeX === #
def format_polynomial_as_latex(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if np.isclose(coeff, 0):
            continue
        sign = "+" if coeff > 0 else "-"
        coeff_str = f"{abs(coeff):.2f}" if not (np.isclose(abs(coeff), 1) and power != 0) else ""
        variable = f"x^{{{power}}}" if power > 1 else ("x" if power == 1 else "")
        term = (f"{coeff_str}{variable}" if coeff > 0 else f"-{coeff_str}{variable}") if not terms else f" {sign} {coeff_str}{variable}"
        terms.append(term)
    return f"f(x) = {''.join(terms)}"

# === JavaScript for Copy Button === #
def js_copy_to_clipboard(text):
    js_code = f"""
    <script>
        function copyToClipboard() {{
            navigator.clipboard.writeText(`$${text}$$`);
            alert("✅ LaTeX copied to clipboard!");
        }}
    </script>
    <button onclick="copyToClipboard()">📋 Copy LaTeX to Clipboard</button>
    """
    st.components.v1.html(js_code, height=40)

# === Generation Process === #
if st.session_state.is_generating and not st.session_state.generation_done:
    with st.spinner("🔄 Generating polynomial... Please wait."):
        try:
            if model_choice == "VAE":
                poly = generate_poly_with_fallback(
                    df_all, requested_degree, epochs=200, penalty_weight=1.0, csv_file=CSV_FILE
                )
                append_poly_to_csv(poly, requested_degree, CSV_FILE)
                poly_latex = format_polynomial_as_latex(poly)
                st.subheader("Generated Polynomial Equation:")
                st.latex(f"{poly_latex}")
                st.write(f"Symbolic Positivity Check: {'Positive ✅' if is_always_positive(poly) else 'Not Positive ❌'}")
                js_copy_to_clipboard(poly_latex)
            else:
                failure_count = st.session_state.gan_failures.get(requested_degree, 0)
                while True:
                    poly = generate_poly_with_gan(requested_degree)
                    is_positive = is_always_positive(poly)

                    failure_count += 1
                    st.session_state.gan_failures[requested_degree] = failure_count

                    if not is_positive and failure_count <= 3:
                        poly_latex = format_polynomial_as_latex(poly)
                        st.subheader("Generated Polynomial Equation:")
                        st.latex(f"{poly_latex}")
                        st.write("Symbolic Positivity Check: Not Positive ❌")
                        js_copy_to_clipboard(poly_latex)
                        st.warning(f"❌ Not Positive Polynomial. Retrying... (Attempt {failure_count})")

                    if is_positive:
                        poly_latex = format_polynomial_as_latex(poly)
                        st.subheader("Generated Polynomial Equation:")
                        st.latex(f"{poly_latex}")
                        st.write("Symbolic Positivity Check: Positive ✅")
                        js_copy_to_clipboard(poly_latex)
                        break

            st.success(f"✅ Polynomial of degree {requested_degree} generated using {model_choice}!")
        except RuntimeError as e:
            st.error(f"❌ Error generating polynomial: {e}")
    st.session_state.is_generating = False
    st.session_state.generation_done = True

