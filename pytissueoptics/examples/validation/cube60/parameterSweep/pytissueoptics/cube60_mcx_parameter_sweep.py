import os
import subprocess
import json
import shutil
import re
from dataclasses import dataclass, asdict
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np


@dataclass
class SweepSimResult:
    u_a: float
    u_s: float
    g: float
    absorbance: float


def update_mcx_input_json(file_path: str, u_a: float, u_s: float, g: float) -> None:
    with open(file_path, 'r') as file:
        data = json.load(file)
    data["Domain"]["Media"][1]["mua"] = u_a
    data["Domain"]["Media"][1]["mus"] = u_s
    data["Domain"]["Media"][1]["g"] = g
    with open(file_path, 'w') as file:
        json.dump(data, file)


def run_mcx_simulation(file_path: str) -> str:
    result = subprocess.run(["mcxcl.exe", "--input", file_path], capture_output=True, text=True)
    return result.stdout


def extract_absorbance_from_mcx_output(output: str) -> float:
    clean_output = re.sub(r'\x1b\[.*?m', '', output)  # Remove ANSI escape codes
    for line in clean_output.splitlines():
        if "absorbed:" in line:
            return float(line.split("absorbed:")[1].strip().rstrip('%'))
    return None


def simulate_and_collect_results(u_a: float, u_s: float, g: float, base_json_path: str, results: List[SweepSimResult],
                                 lock: Lock) -> None:
    json_file_path = f"./jsonTestFiles/input_{u_a}_{u_s}_{g}.json"
    shutil.copy(base_json_path, json_file_path)
    update_mcx_input_json(json_file_path, u_a, u_s, g)

    output = run_mcx_simulation(json_file_path)
    absorbance = extract_absorbance_from_mcx_output(output)

    with lock:
        if absorbance is not None:
            results.append(SweepSimResult(u_a, u_s, g, absorbance))
        else:
            results.append(SweepSimResult(u_a, u_s, g, 0.0))


def main(g_values: List[float], u_a_values: List[float], u_s_values: List[float]) -> List[SweepSimResult]:
    results = []
    lock = Lock()
    os.makedirs("jsonTestFiles", exist_ok=True)

    base_json_path = "cube60custom.json"

    with ThreadPoolExecutor() as executor:
        futures = []
        for u_a in u_a_values:
            for u_s in u_s_values:
                for g in g_values:
                    futures.append(
                        executor.submit(simulate_and_collect_results, u_a, u_s, g, base_json_path, results, lock))
                    print(f"Submitted simulation for u_a={u_a}, u_s={u_s}, g={g}")

        for future in as_completed(futures):
            future.result()

    return results


def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump([asdict(result) for result in results], f, indent=4)

def load_results_from_json(filename):
    with open(filename, 'r') as f:
        results_dicts = json.load(f)
    return [SweepSimResult(**result) for result in results_dicts]


if __name__ == "__main__":
    g_values = [1, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.3, 0]
    u_a_values = [0.005]  # Example list of u_a values
    u_s_values = [0.005, 0.01, 0.015, 0.020, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.060, 0.065, 0.07, 0.075,
                  0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7,
                  0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 10, 15, 20, 25, 30]

    results = main(g_values, u_a_values, u_s_values)

    for result in results:
        print(result)

    save_results_to_json(results, 'cube60_sweep_mcx_results.json')
