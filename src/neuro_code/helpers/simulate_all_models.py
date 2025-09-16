import pandas as pd
from neuro_code.helpers.model_configs from neuro_code.helpers import model_configs
from neuro_code.helpers.simulate_model_range import simulate_model


def simulate_all_models_and_save():
    all_dfs = []

    for model_name in model_configs.keys():
        print(f"Simulating model: {model_name}")
        model_df = simulate_model(model_name)
        all_dfs.append(model_df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv("all_model_predictions.csv", index=False)
    print("All model simulations complete. Saved to all_model_predictions.csv")


if __name__ == '__main__':
    simulate_all_models_and_save()
