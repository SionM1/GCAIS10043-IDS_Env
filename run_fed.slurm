#!/bin/bash
#SBATCH --job-name=FedAvgGPU
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=6       # 1 server + 10 clients
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=fedavg_%j.out
#SBATCH --error=fedavg_%j.err

# 1) Load the modules needed
module load python/3.10.4

# 2) Activate your Python environment
source /home/c.c24079993/fed_env/bin/activate

# 3) Move to your code directory (adjust path as needed)
cd /home/c.c24079993/fed_env/

# This will produce client_data/client_0.csv etc.
python partition_dataset.py

# 4) Launch the Flower server on one task
srun --exclusive -n1 python server.py &

# 5) Give server a few seconds to start
sleep 5

# 6) Launch the five clients on separate tasks
for i in {0..9}; do
    srun --exclusive -n1 python client.py "$i" &
done

# 7) Wait until all background processes finish
wait

