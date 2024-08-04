import os
import sys
import argparse

from pytissueoptics import __version__
from pytissueoptics.examples import loadExamples

ap = argparse.ArgumentParser(prog=f"{sys.executable} -m pytissueoptics", description="Run PyTissueOptics examples. ")
ap.add_argument("-v", "--version", action="version", version=f"PyTissueOptics {__version__}")
ap.add_argument("-e", "--examples", required=False, default="all",
                help="Run specific examples by number, e.g. -e 1,2,3. ")
ap.add_argument("-l", "--list", required=False, action="store_true", help="List available examples. ")
ap.add_argument("-t", "--tests", required=False, action="store_true", help="Run unit tests. ")

args = vars(ap.parse_args())
runExamples = args["examples"]
runTests = args["tests"]
listExamples = args["list"]
allExamples = loadExamples()

if listExamples:
    print("Available examples:")
    for i, example in enumerate(allExamples):
        print(f"{i + 1}. {example.name}.py {example.title}")
    sys.exit()

if runTests:
    moduleDir = os.path.dirname(__file__)
    err = os.system(f"{sys.executable} -m unittest discover -s {moduleDir} -p 'test*.py'")
    sys.exit(err)

if runExamples == "all":
    runExamples = list(range(1, len(allExamples) + 1))
elif runExamples:
    runExamples = [int(i) for i in runExamples.split(",")]
else:
    print("No examples specified. Use -e 1,2,3 to run specific examples. ")
    sys.exit()

print(f"Running examples {runExamples}...")
for i in runExamples:
    example = allExamples[i - 1]
    print(f"\nExample {i}: {example.name}.py")
    print(f"TITLE: {example.title}")
    print(f"DESCRIPTION: {example.description}")
    print("\n--------------- Begin source code ---------------")
    print(example.sourceCode)
    print("---------------- End source code ----------------")
    print("\n----------------- Begin output ------------------")
    example.func()
    print("------------------ End output -------------------")
