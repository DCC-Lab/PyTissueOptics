from dataclasses import dataclass, asdict
from typing import List, Dict
import json
import os

g_values = [1, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.3, 0]
mu_a_values = [0.005]
mu_s_values = [0.005, 0.01, 0.015, 0.020, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.060, 0.065, 0.07, 0.075,
              0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7,
              0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 10, 15, 20, 25, 30]

@dataclass
class SweepSimResult:
    mua: float
    mus: float
    g: float
    absorbance: float


    @classmethod
    def save_to_json(cls, results: List['SweepSimResult'], filename: str, software="mcx"):
        """
        Saves a list of SweepSimResult instances to a JSON file in a hierarchical format.
        If the file already exists, it appends the new data under the software key.

        :param results: List of SweepSimResult instances to save.
        :param filename: Filename to save the JSON data.
        :param software: The simulation software name to use as a key in the JSON file.
        """
        # Load existing data if the file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Populate the nested dictionary structure
        for result in results:
            mua = str(result.mua)
            mus = str(result.mus)
            g = str(result.g)

            # Ensure hierarchy for mua, mus, and g levels
            if mua not in data:
                data[mua] = {}
            if mus not in data[mua]:
                data[mua][mus] = {}
            if g not in data[mua][mus]:
                data[mua][mus][g] = {}

            # Set the absorbance for the specified software, updating only this key if it exists
            data[mua][mus][g][software] = result.absorbance

        # Write the updated data back to the file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_json(cls, filename: str, software: str = "mcx") -> List['SweepSimResult']:
        """
        Loads a list of SweepSimResult instances from a hierarchical JSON file for a specified software.

        :param filename: Filename to load the JSON data from.
        :param software: Name of the software to retrieve absorbance data for.
        :return: List of SweepSimResult instances.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        # Flatten the hierarchical structure into a list of SweepSimResult instances
        results = []
        for mua, mus_values in data.items():
            for mus, g_values in mus_values.items():
                for g, absorbance_dict in g_values.items():
                    if software in absorbance_dict:
                        results.append(
                            cls(
                                g=float(g),
                                mus=float(mus),
                                absorbance=float(absorbance_dict[software]),
                                mua=float(mua)
                            )
                        )
        return results