import os
import subprocess
import json
import shutil
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from utils import SweepSimResult, g_values, mu_a_values, mu_s_values


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
    json_file_path = f"temp/jsonTestFiles/input_{u_a}_{u_s}_{g}.json"
    shutil.copy(base_json_path, json_file_path)
    update_mcx_input_json(json_file_path, u_a, u_s, g)

    output = run_mcx_simulation(json_file_path)
    absorbance = extract_absorbance_from_mcx_output(output)

    with lock:
        if absorbance is not None:
            results.append(SweepSimResult(u_a, u_s, g, absorbance))
        else:
            results.append(SweepSimResult(u_a, u_s, g, 0.0))


def mcx_cube60_sweep_simulator(u_a_values: List[float], u_s_values: List[float], g_values: List[float]) -> List[SweepSimResult]:
    results = []
    lock = Lock()
    os.makedirs("../temp/jsonTestFiles", exist_ok=True)

    base_json_path = "cube60customFresnel_mcxInput.json"

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


if __name__ == "__main__":
    results = mcx_cube60_sweep_simulator([0.3], [20], [0.7])
    SweepSimResult.save_to_json(results, '../cube60_mcx_n=1.0_results.json')
