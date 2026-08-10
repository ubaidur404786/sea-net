# H100 94 GiB, 6 nodes × 2 GPUs — best availability
oarsub -q abaca -I -l gpu=1,walltime=12:00:00 -p musa
oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "musa"



# A100 80 GiB
oarsub -q abaca -I -l gpu=1,walltime=12:00:00 -p esterel37
oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "esterel37"

# RTX A6000 48 GiB
oarsub -q abaca -I -l gpu=1,walltime=8:00:00 -p esterel17 not working 

oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p esterel17


# 94 GB H100
oarsub -q abaca -I -l gpu=1,walltime=4:00:00 -p "gpu_model LIKE '%H100%'"

# 45 GB L40S Besteffort only 
oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "gpu_model LIKE '%L40S%'"


oarsub -q abaca -I -l gpu=1,walltime=4:00:00 -p "esterel35 or esterel25 or esterel26"

oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "esterel35 or esterel40 or esterel43 or esterel37 or esterel17"


# 23 GB
oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "esterel23 or esterel22"



oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "esterel17 or esterel22"

# 32 GB

oarsub -q besteffort -I -l gpu=1,walltime=12:00:00 -p "esterel28 or esterel30"