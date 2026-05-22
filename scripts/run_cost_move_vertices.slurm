#!/bin/bash
#SBATCH --job-name=vertax_cost_mv
#SBATCH --output=logs/%x.o%j
#SBATCH --error=logs/%x.e%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_short

set -euo pipefail

# --- environment ---------------------------------------------------------
module purge
module load anaconda3/2023.09-0/none-none
# shellcheck disable=SC1091
source activate vertax

# --- runtime config ------------------------------------------------------
WORKDIR="/gpfs/users/pasquial/vertax_project/VertAX"
cd "${WORKDIR}"

mkdir -p logs
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="${WORKDIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# ott-jax>=0.6 needs jax>=0.4.36 (register_dataclass without explicit fields)
pip install -q 'jax>=0.4.36' 'jaxlib>=0.4.36'

echo "[$(date -Is)] Running cost_move_vertices on $(hostname)"
echo "  workdir=${WORKDIR}"
echo "  python=$(which python)"
echo "  cpus=${SLURM_CPUS_PER_TASK:-unknown}"

srun --unbuffered python -m vertax.cost_move_vertices

echo "[$(date -Is)] Done."
