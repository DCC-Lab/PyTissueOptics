import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_comparison(data, model1, model2, plot_type, mua=None, mus=None, g=None):
    """
    Plot comparisons between two models based on plot_type.

    Parameters:
    - data: dict, the nested data dictionary containing absorbance values.
    - model1, model2: str, names of the models to compare.
    - plot_type: str, one of ['absolute', 'absolute_difference', 'relative_error', 'all', 'three'].
    - mua, mus, g: float, specify one or two parameters to hold constant.
    """
    if sum(x is not None for x in [mua, mus, g]) != 1 and sum(x is not None for x in [mua, mus, g]) != 2:
        raise ValueError("Exactly one or two of mua, mus, or g must be specified.")

        # Parameter labels
    param_labels = {
        'mua': r"Absorption Coefficient $\mu_a$ (cm$^{-1}$)",
        'mus': r"Scattering Coefficient $\mu_s$ (cm$^{-1}$)",
        'g': "Anisotropy Factor g"
    }

    # Determine the fixed parameter(s)
    if mua is not None:
        fixed_param, fixed_value = 'mua', str(mua)
        x_param, y_param = 'mus', 'g'
        x_label, y_label = param_labels[x_param], param_labels[y_param]
        x_values = sorted(map(float, data[fixed_value].keys()))
        y_values = sorted({g for mus_values in data[fixed_value].values() for g in mus_values.keys()})
    elif mus is not None:
        fixed_param, fixed_value = 'mus', str(mus)
        x_param, y_param = 'mua', 'g'
        x_label, y_label = param_labels[x_param], param_labels[y_param]
        x_values = sorted(map(float, data.keys()))
        y_values = sorted(
            {g for mua_values in data.values() if fixed_value in mua_values for g in mua_values[fixed_value].keys()})
    else:
        fixed_param, fixed_value = 'g', str(g)
        x_param, y_param = 'mua', 'mus'
        x_label, y_label = param_labels[x_param], param_labels[y_param]
        x_values = sorted(map(float, data.keys()))
        y_values = sorted(
            {mus for mua_values in data.values() for mus_values in mua_values.values() if fixed_value in mus_values})

    # Prepare data matrices
    model1_data = np.zeros((len(x_values), len(y_values)))
    model2_data = np.zeros((len(x_values), len(y_values)))
    abs_diff_data = np.zeros((len(x_values), len(y_values)))
    rel_error_data = np.zeros((len(x_values), len(y_values)))

    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            try:
                if fixed_param == 'mua':
                    m1_val = data[fixed_value][str(x_val)][y_val].get(model1, 0)
                    m2_val = data[fixed_value][str(x_val)][y_val].get(model2, 0)
                elif fixed_param == 'mus':
                    m1_val = data[str(x_val)][fixed_value][y_val].get(model1, 0)
                    m2_val = data[str(x_val)][fixed_value][y_val].get(model2, 0)
                elif fixed_param == 'g':
                    m1_val = data[str(x_val)][y_val][fixed_value].get(model1, 0)
                    m2_val = data[str(x_val)][y_val][fixed_value].get(model2, 0)

                model1_data[i, j] = m1_val
                model2_data[i, j] = m2_val
                abs_diff_data[i, j] = abs(m1_val - m2_val)
                rel_error_data[i, j] = (
                            (m1_val - m2_val) / m1_val * 100) if m1_val != 0 else np.nan  # Convert to percentage
            except KeyError:
                model1_data[i, j] = model2_data[i, j] = abs_diff_data[i, j] = rel_error_data[i, j] = np.nan

    # Determine plot layout based on plot_type
    if plot_type == 'absolute':
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        sns.heatmap(model1_data, xticklabels=y_values, yticklabels=x_values, ax=axes[0], cmap='viridis',
                    cbar_kws={'label': '(%)'})
        axes[0].set_title(f'Absorbance ({model1})')
        axes[0].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(model2_data, xticklabels=y_values, yticklabels=x_values, ax=axes[1], cmap='viridis',
                    cbar_kws={'label': '(%)'})
        axes[1].set_title(f'Absorbance ({model2})')
        axes[1].collections[0].colorbar.ax.set_title("(%)")

    elif plot_type == 'absolute_difference':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(abs_diff_data, xticklabels=y_values, yticklabels=x_values, ax=ax, cmap='magma',
                    cbar_kws={'label': '(%)'})
        ax.set_title(f'Absolute Difference |{model1} - {model2}|')
        ax.collections[0].colorbar.ax.set_title("(%)")

    elif plot_type == 'relative_error':
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(rel_error_data, xticklabels=y_values, yticklabels=x_values, ax=ax, cmap='coolwarm',
                    cbar_kws={'label': '(%)'}, annot=False, fmt=".0f")
        ax.set_title(f'Relative Error between {model1} and {model2}')
        ax.collections[0].colorbar.ax.set_title("(%)")

    elif plot_type == 'three':
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        sns.heatmap(model1_data, xticklabels=y_values, yticklabels=x_values, ax=axes[0], cmap='viridis')
        axes[0].set_xlabel(y_label)
        axes[0].set_ylabel(x_label)
        axes[0].set_title(f'Absorbance ({model1})')
        axes[0].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(abs_diff_data, xticklabels=y_values, yticklabels=x_values, ax=axes[1], cmap='magma')
        axes[1].set_xlabel(y_label)
        axes[1].set_ylabel(x_label)
        axes[1].set_title(f'Absolute Difference |{model1} - {model2}|')
        axes[1].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(rel_error_data, xticklabels=y_values, yticklabels=x_values, ax=axes[2], cmap='coolwarm')
        axes[2].set_xlabel(y_label)
        axes[2].set_ylabel(x_label)
        axes[2].set_title(f'Relative Error between {model1} and {model2}')
        axes[2].collections[0].colorbar.ax.set_title("(%)")

    elif plot_type == 'all':
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
        sns.heatmap(model1_data, xticklabels=y_values, yticklabels=x_values, ax=axes[0], cmap='viridis')
        axes[0].set_xlabel(y_label)
        axes[0].set_ylabel(x_label)
        axes[0].set_title(f'Absorbance ({model1})')
        axes[0].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(model2_data, xticklabels=y_values, yticklabels=x_values, ax=axes[1], cmap='viridis')
        axes[1].set_xlabel(y_label)
        axes[1].set_ylabel(x_label)
        axes[1].set_title(f'Absorbance ({model2})')
        axes[1].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(abs_diff_data, xticklabels=y_values, yticklabels=x_values, ax=axes[2], cmap='magma')
        axes[2].set_xlabel(y_label)
        axes[2].set_ylabel(x_label)
        axes[2].set_title(f'Absolute Difference |{model1} - {model2}|')
        axes[2].collections[0].colorbar.ax.set_title("(%)")

        sns.heatmap(rel_error_data, xticklabels=y_values, yticklabels=x_values, ax=axes[3], cmap='coolwarm',
                    annot=False, fmt=".0f")
        axes[3].set_xlabel(y_label)
        axes[3].set_ylabel(x_label)
        axes[3].set_title(f'Relative Error between {model1} and {model2}')
        axes[3].collections[0].colorbar.ax.set_title("(%)")

    for ax in fig.axes:
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    plt.subplots_adjust(wspace=0.4)
    plt.suptitle(
        f"Comparison between {model1} and {model2} ({plot_type.replace('_', ' ').capitalize()})\nFixed {fixed_param} = {fixed_value}")
    plt.show()


# Load data and example usage
data = load_data('cube60_sweep_results.json')
plot_comparison(data, 'pytissueoptics', 'mcx', plot_type='three', mua=0.005)  # New option for three plots
