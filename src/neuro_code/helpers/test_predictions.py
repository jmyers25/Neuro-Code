import pandas as pd
import numpy as np
from neuro_code.helpers.get_predicted_df from neuro_code.helpers import get_predicted_df


def test_prediction_output_structure():
    # Create a  fake dataset
    data = pd.DataFrame({
        'Trial_type': [1, 2, 3, 4],
        'Response': [1, 2, 1, 2],
        'Points_earned': [100, -10, 495, 0]
    })

    # Define a minimal set of parameters for the symmetric model
    pars_dict = {
        'alpha': 0.5,
        'exp': 1.0,
        'beta': 0.5,
        'lossave': 1.0
    }

    try:
        result = get_predicted_df(data, pars_dict)
        required_columns = ['Trial_type', 'Response', 'Points_earned', 'EV', 'PE',
                            'choiceprob', 'pred_choice', 'pred_reward']

        missing = [col for col in required_columns if col not in result.columns]
        assert not missing, f"Missing required columns: {missing}"

        assert len(result) == len(data), "Row count mismatch between input and output."

        print("✅ test_prediction_output_structure passed.")
    except Exception as e:
        print(f"❌ test_prediction_output_structure failed: {e}")


if __name__ == '__main__':
    test_prediction_output_structure()