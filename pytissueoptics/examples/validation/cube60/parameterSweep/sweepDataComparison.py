
def plot_absorbance(results, g_value=None, mus_value=None):
    x = []
    y = []
    for result in results:
        if g_value is not None and result.g == g_value:
            x.append(result.mus)
            y.append(result.absorbance)
        elif mus_value is not None and result.mus == mus_value:
            x.append(result.g)
            y.append(result.absorbance)

    plt.plot(x, y, 'o-')
    plt.xlabel('g' if g_value is None else 'mus')
    plt.ylabel('Total Energy Absorbed (%)')
    plt.show()


def compute_absorbance_difference(
        results_a: List[SweepSimResult],
        results_b: List[SweepSimResult],
        g_value: Optional[float] = None,
        mus_value: Optional[float] = None
) -> Tuple[List[float], List[float]]:
    if g_value is None and mus_value is None:
        raise ValueError("Either g_value or mus_value must be specified.")

    # Determine the filtering criterion and dependent variable
    if g_value is not None:
        # Filter by g_value, making mus the dependent variable
        filtered_a = [result for result in results_a if result.g == g_value]
        filtered_b = [result for result in results_b if result.g == g_value]
        dependent_attr = 'mus'
    else:
        # Filter by mus_value, making g the dependent variable
        filtered_a = [result for result in results_a if result.mus == mus_value]
        filtered_b = [result for result in results_b if result.mus == mus_value]
        dependent_attr = 'g'

    # Organize results by the dependent attribute
    results_a_dict = {getattr(result, dependent_attr): result.absorbance for result in filtered_a}
    results_b_dict = {getattr(result, dependent_attr): result.absorbance for result in filtered_b}

    # Calculate the absorbance differences for matching dependent values
    dependant_values = []
    delta_absorbance = []

    for value in results_a_dict:
        if value in results_b_dict:
            dependant_values.append(value)
            delta_absorbance.append(results_b_dict[value] - results_a_dict[value])

    return dependant_values, delta_absorbance


def plot_absorbance_difference(dependant_values: List[float], delta_absorbance: List[float], dependent_attr: str):
    """
    Plots the absorbance difference against the dependent variable.

    :param dependant_values: List of mus or g values (dependent variable).
    :param delta_absorbance: List of absorbance differences for the dependent variable values.
    :param dependent_attr: Name of the dependent attribute ('mus' or 'g') for labeling.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(dependant_values, delta_absorbance, marker='o', linestyle='-', color='b', label="Δ Absorbance")
    plt.xlabel(f"{dependent_attr} value")
    plt.ylabel("Absorbance Difference")
    plt.title(f"Absorbance Difference vs {dependent_attr.capitalize()}")
    plt.grid(True)
    plt.legend()
    plt.show()