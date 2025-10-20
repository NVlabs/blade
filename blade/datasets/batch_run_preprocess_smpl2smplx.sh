# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# Input arguments
ENV_NAME=$1
DATASET=$2
MACHINE_RANK=$3
NUM_PROCESSES_PER_MACHINE=$4
SEQ_PER_PROCESS=$5
export EXP_NAME=asd

source ~/.bashrc

#source ~/.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
HOME_DIR=~
__conda_setup="$('~/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${HOME_DIR}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME_DIR}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${HOME_DIR}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda deactivate; conda activate $ENV_NAME

export TORCH_HOME=~/.cache/
export WANDB_DIR=~/.wandb_cache/
#export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=12
echo $ENV_NAME

conda env list
conda list

# Usage: ./run_on_machine.sh <machine_rank> <num_processes_per_machine> <num_sequences_per_process>

if [ $# -ne 5 ]; then
  echo "Usage: $0 <conda env> <dataset> <machine_rank> <num_processes_per_machine> <num_sequences_per_process>"
  exit 1
fi

# bash batch_run_preprocess_ourdata.sh <machine_rank> 50 5

# Get the list of available CPUs
AVAILABLE_CPUS=$(taskset -cp $$ | grep -oP '(?<=list: ).*')
IFS=',' read -ra CPU_RANGES <<< "$AVAILABLE_CPUS"  # Split by comma if there are multiple ranges
CPU_LIST=()

for range in "${CPU_RANGES[@]}"; do
    if [[ "$range" == *"-"* ]]; then
        START=$(echo "$range" | cut -d'-' -f1)
        END=$(echo "$range" | cut -d'-' -f2)
        for ((i=START; i<=END; i++)); do
            CPU_LIST+=($i)
        done
    else
        CPU_LIST+=($range)
    fi
done
echo "${CPU_LIST[@]}"

# Check if enough CPUs are available
NUM_CPUS=${#CPU_LIST[@]}
if [ $NUM_CPUS -lt $NUM_PROCESSES_PER_MACHINE ]; then
  echo "Not enough CPUs available to run $NUM_PROCESSES_PER_MACHINE processes. Only $NUM_CPUS CPUs available."
  exit 1
fi

# Function to run a process for a given set of sequences
run_process() {
  DATASET=$1
  CPU=$2
  PROCESS_ID=$3
  SEQ_START=$4
  SEQ_END=$5
  LOG_FILE="process_${PROCESS_ID}_${SEQ_START}_${SEQ_END}.log"
  CMD="taskset -c $CPU python preprocess_smpldata_chunked.py $DATASET $PROCESS_ID $SEQ_START $SEQ_END"

  echo "$CMD"

  echo "Running command: $CMD" > $LOG_FILE

  # Use taskset to bind the process to a single CPU
  $CMD >> $LOG_FILE 2>&1 &
}

# Main loop to spawn processes
for ((i=0; i<NUM_PROCESSES_PER_MACHINE; i++)); do
  CPU=${CPU_LIST[$i]}
  PROCESS_ID=$((MACHINE_RANK * NUM_PROCESSES_PER_MACHINE + i))

  SEQ_START=$((PROCESS_ID * SEQ_PER_PROCESS))
  SEQ_END=$(((PROCESS_ID+1) * SEQ_PER_PROCESS - 1))


  run_process $DATASET $CPU $PROCESS_ID $SEQ_START $SEQ_END
done

# Wait for all background processes to complete
wait

echo "All processes on machine $MACHINE_RANK completed."
