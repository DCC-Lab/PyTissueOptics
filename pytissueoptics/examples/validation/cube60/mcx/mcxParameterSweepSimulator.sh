#!/bin/bash

# Default values for parameters
u_a_start=0.005
u_a_end=5
u_a_step=0

u_s_start=0.01
u_s_end=10
u_s_step=0

g_start=0.1
g_end=1
g_step=0

# Help function to explain the usage of the script
help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -a u_a_start      Start value for u_a (default: 0.005)"
  echo "  -b u_a_end        End value for u_a (default: 5)"
  echo "  -c u_a_step       Step value for u_a (default: 0, use 0 to avoid sweeping)"
  echo "  -d u_s_start      Start value for u_s (default: 0.01)"
  echo "  -e u_s_end        End value for u_s (default: 10)"
  echo "  -f u_s_step       Step value for u_s (default: 0, use 0 to avoid sweeping)"
  echo "  -g g_start        Start value for g (default: 0.1)"
  echo "  -h g_end          End value for g (default: 1)"
  echo "  -i g_step         Step value for g (default: 0, use 0 to avoid sweeping)"
  echo "  --help            Display this help message and exit"
  exit 0
}

# Parse command-line options
while getopts "a:b:c:d:e:f:g:h:i:-:" opt; do
  case $opt in
    a) u_a_start=$OPTARG ;;
    b) u_a_end=$OPTARG ;;
    c) u_a_step=$OPTARG ;;
    d) u_s_start=$OPTARG ;;
    e) u_s_end=$OPTARG ;;
    f) u_s_step=$OPTARG ;;
    g) g_start=$OPTARG ;;
    h) g_end=$OPTARG ;;
    i) g_step=$OPTARG ;;
    -)
      case $OPTARG in
        help) help ;;
        *) echo "Invalid option --$OPTARG" >&2; exit 1 ;;
      esac
      ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Ensure the directory exists
mkdir -p ./jsonTestFiles

# Results file
results_file="./jsonTestFiles/results.txt"
echo "u_a u_s g absorbed" > $results_file

# Function to update the json file with new parameters
update_json() {
  local ua=$1
  local us=$2
  local g=$3
  local file=$4
  jq ".Domain.Media[1].mua = $ua | .Domain.Media[1].mus = $us | .Domain.Media[1].g = $g" $file > temp.json && mv temp.json $file
}

# Function to check if we should sweep or use a single value
generate_sequence() {
  local start=$1
  local step=$2
  local end=$3
  if [[ "$step" == "0" ]]; then
    echo $start
  else
    seq $start $step $end
  fi
}

# Sweep through parameters
for u_a in $(generate_sequence $u_a_start $u_a_step $u_a_end); do
  for u_s in $(generate_sequence $u_s_start $u_s_step $u_s_end); do
    for g in $(generate_sequence $g_start $g_step $g_end); do
      # Copy the base JSON file and modify it
      json_file="./jsonTestFiles/input_${u_a}_${u_s}_${g}.json"
      cp cube60custom.json $json_file
      update_json $u_a $u_s $g $json_file

      # Run mcxcl with the updated JSON file
      output=$(mcxcl.exe --input $json_file)
      
      # Extract the absorbed percentage from the output
      clean_output=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')
      absorbed=$(echo "$clean_output" | grep -oP 'absorbed: \K[0-9]+\.[0-9]+%?')

      # Write the parameters and absorbed percentage to the results file
      if [[ ! -z "$absorbed" ]]; then
        echo "$u_a $u_s $g $absorbed" >> $results_file
      else
        echo "$u_a $u_s $g NO DATA" >> $results_file
      fi
    done
  done
done

echo "Sweep complete. Results stored in $results_file"
