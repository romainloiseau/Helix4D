import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_mem", default=32, choices=[16, 32], type=int)
parser.add_argument("--hours", default=100, type=int)
parser.add_argument("--n_gpus", default=1, type=int)
parser.add_argument("--batch_size", default=2, type=int)
args = parser.parse_args()

def overide_config(base, dico):
    return base + " ".join([f"{k}=" + str(v) for k, v in dico.items()]) + " "

default_config =  {
    "model.data.num_workers": min(4*args.n_gpus, 16),
    "model.data.batch_size": args.batch_size*args.n_gpus,
    "profile": True
}
if args.n_gpus != 1:
    default_config["trainer.gpus"] = -1,
    default_config["+trainer.accelerator"] = "dp"

slurm = overide_config(f"""#!/bin/bash
#SBATCH --job-name=GPU # nom du job
#SBATCH --output=GPU_%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=GPU_%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time={min(args.hours-1, 99)}:59:00 # temps maximal d’allocation        (19:59:00 on standard qos_gpu-t3 partition)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 1 taches (ou processus MPI)
#SBATCH --gres=gpu:{args.n_gpus} # reserver 1 GPU
#SBATCH --cpus-per-task={min(10*args.n_gpus, 20)} # reserver 10 CPU par tache (et memoire associee (4*10G))
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --qos=qos_gpu-t{4 if args.hours>20 else 3}      #qos_gpu-t4 for long runs (>20h), qos_gpu-t3 either

module purge # nettoyer les modules herites par defaut
module load pytorch-gpu/py3/1.10.0 # charger les modules
set -x # activer l’echo des commandes

srun python main.py """, default_config)

models = sum([[[
    {
        "+data": "semantic-kitti",
        "+experiment": xp,
        "data.blocks_per_rotation": blocks_per_rotation,
        "hydra.run.dir": f"outputs/{xp}/ABLATION_B{blocks_per_rotation}"
    }
] for xp in ["ours", "ours_noposenc", "ours_C3D", "ours_KdiffQ", "ours_nopast", "ours_notrans", "ours_tiny"]
] for blocks_per_rotation in [5, 1]
], [])
    
print(f"##### Launching {len(models)} jobs #####")
for i, model in enumerate(models):    
    f = open("auto.slurm", "w")
    f.write(overide_config(slurm, model))
    f.close()
    os.system(f"sbatch -C v100-{args.gpu_mem}g auto.slurm")
    os.remove("auto.slurm")