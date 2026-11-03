#!/bin/bash
# Run from the project root, after loading the course compiler/MPI environment.
set -euo pipefail

# Submit/collect can be invoked from any directory. Slurm runs its spool copy
# from the project root specified by --chdir below.
if [ "${1:-submit}" != run ]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

case "${1:-submit}" in
submit)
    # Compile on the cluster with the same optimization level for every solver.
    make -B all CXXFLAGS=-O3
    mkdir -p results
    options=(--parsable --partition="${PARTITION:-cmsc5702_hpc}"
             --qos="${QOS:-cmsc5702}" --time="${WALLTIME:-01:00:00}"
             --exclusive --hint=nomultithread --chdir="$PWD")
    sbatch "${options[@]}" --job-name=heat_seq --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --output=results/seq-%j.log "$PWD/scripts/benchmark.sh" run seq 1
    for p in 2 4 8 16 32; do
        sbatch "${options[@]}" --job-name="heat_omp_$p" --nodes=1 --ntasks=1 \
            --cpus-per-task="$p" --output="results/omp_${p}-%j.log" \
            "$PWD/scripts/benchmark.sh" run omp "$p"
        case "$p" in
            2) nodes=1; per_node=2 ;;
            4) nodes=1; per_node=4 ;;
            8) nodes=2; per_node=4 ;;
            16) nodes=4; per_node=4 ;;
            32) nodes=4; per_node=8 ;;
        esac
        sbatch "${options[@]}" --job-name="heat_mpi_$p" --nodes="$nodes" --ntasks="$p" \
            --ntasks-per-node="$per_node" --cpus-per-task=1 \
            --output="results/mpi_${p}-%j.log" "$PWD/scripts/benchmark.sh" run mpi "$p"
    done
    ;;
run)
    mode=$2
    p=$3
    output="results/${mode}_${p}.csv"
    # Keep each job's rows separate until all jobs have completed.
    printf 'n,p,runtime1,runtime2,runtime3,runtime4,runtime5\n' > "$output.part"
    case "$mode" in
        seq) command=(srun --ntasks=1 --cpu-bind=cores ./build/heat_seq) ;;
        omp)
            export OMP_NUM_THREADS="$p" OMP_DYNAMIC=FALSE
            export OMP_PLACES=cores OMP_PROC_BIND=close
            command=(srun --ntasks=1 --cpu-bind=cores ./build/heat_omp)
            ;;
        mpi)
            per_node=$((p / SLURM_JOB_NUM_NODES))
            hostfile="results/hosts_${SLURM_JOB_ID}.txt"
            scontrol show hostnames "$SLURM_JOB_NODELIST" | \
                awk -v slots="$per_node" '{print $0 " slots=" slots}' > "$hostfile"
            # The course helper excludes the Docker network for Open MPI.
            export OMPI_MCA_btl_tcp_if_exclude=lo,docker0
            command=("${MPIEXEC:-mpiexec.openmpi}" --hostfile "$hostfile"
                     --map-by "ppr:${per_node}:node" --bind-to core -n "$p" ./build/heat_mpi)
            ;;
    esac
    for n in 200 300 400; do
        printf '%s,%s' "$n" "$p" >> "$output.part"
        for run in 1 2 3 4 5; do
            log="results/${mode}_${n}_${p}_${run}.log"
            "${command[@]}" "$n" "$n" 100000 "results/${mode}_${n}_${p}.txt" | tee "$log"
            # The program reports only find_steady_state(), in seconds.
            elapsed=$(awk '/^Elapsed time = / {print $4}' "$log")
            printf ',%s' "$elapsed" >> "$output.part"
        done
        printf '\n' >> "$output.part"
    done
    mv "$output.part" "$output"
    ;;
collect)
    for mode in seq omp mpi; do
        counts="2 4 8 16 32"
        if [ "$mode" = seq ]; then counts=1; fi
        printf 'n,p,runtime1,runtime2,runtime3,runtime4,runtime5\n' > "data/timings/${mode}_times.csv.part"
        for p in $counts; do
            tail -n +2 "results/${mode}_${p}.csv" >> "data/timings/${mode}_times.csv.part"
        done
        mv "data/timings/${mode}_times.csv.part" "data/timings/${mode}_times.csv"
    done
    ;;
*)
    printf 'Usage: bash scripts/benchmark.sh [submit|collect]\n' >&2
    exit 1
    ;;
esac
