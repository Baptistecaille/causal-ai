import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import warnings

# Suppress some potential warnings for cleaner output
warnings.filterwarnings("ignore")


def create_causal_data(n_samples=1000):
    """
    Creates data where Treatment causes Outcome, with a Confounder affecting both.
    Graph: Confounder -> Treatment -> Outcome
           Confounder -> Outcome
    """
    np.random.seed(42)

    # Confounder (e.g., Age)
    confounder = np.random.normal(0, 1, n_samples)

    # Treatment (e.g., Binary drug assignment)
    # Depends on confounder + random noise
    treatment_prob = 1 / (1 + np.exp(-confounder))  # Sigmoid
    treatment = np.random.binomial(1, treatment_prob)

    # Outcome (e.g., Recovery time)
    # Depends on Treatment (effect = -5) and Confounder (effect = 2)
    outcome = -5 * treatment + 2 * confounder + np.random.normal(0, 1, n_samples)

    # Create DataFrame
    df = pd.DataFrame(
        {"treatment": treatment, "outcome": outcome, "confounder": confounder}
    )

    # Add a binary version of confounder for diversity (optional)
    df["confounder_bin"] = (df["confounder"] > 0).astype(int)

    return df


def main():
    print("----------------------------------------------------------------")
    print("DoWhy Causal Inference Example")
    print("----------------------------------------------------------------")

    # 1. Generate Data
    print("\n1. Generating data...")
    data = create_causal_data()
    print("   True Causal Effect is -5.0")
    print(data.head())

    # 2. Model
    print("\n2. Modeling...")
    # Define a causal model
    model = CausalModel(
        data=data,
        treatment="treatment",
        outcome="outcome",
        common_causes=["confounder"],
        # graph="digraph { confounder -> treatment; confounder -> outcome; treatment -> outcome; }" # Optional if common_causes provided
    )
    model.view_model()  # This is usually for plotting, might not show in terminal but good to have

    # 3. Identify
    print("\n3. Identifying Causal Effect...")
    identified_estimand = model.identify_effect()
    print(identified_estimand)

    # 4. Estimate
    print("\n4. Estimating Causal Effect...")
    # Using Linear Regression
    estimate = model.estimate_effect(
        identified_estimand, method_name="backdoor.linear_regression"
    )
    print("   Estimated Effect: {:.2f}".format(estimate.value))

    # 5. Refute
    print("\n5. Refuting Estimate (Robustness Check)...")
    # Add a random common cause variable
    refute = model.refute_estimate(
        identified_estimand, estimate, method_name="random_common_cause"
    )
    print(refute)


if __name__ == "__main__":
    main()
