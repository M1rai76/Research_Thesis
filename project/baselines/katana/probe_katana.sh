#!/usr/bin/env bash
# probe_katana.sh
# Baseline - Reflexion (Shinn et al., 2023)
# Author : Gurdiraj Bal (z5386590)
#
# READ-ONLY inspection of a UNSW Katana login node. Changes nothing, submits
# nothing, downloads nothing. Run on a Katana LOGIN node and share the output
# so the vLLM serve/run job scripts can be finalised with real site values.

echo "############ WHOAMI / HOST ############"
whoami; echo "HOME=$HOME"; hostname
cat /etc/os-release 2>/dev/null | grep PRETTY_NAME; uname -r

echo; echo "############ SCHEDULER ############"
command -v qsub  >/dev/null && echo "PBS  (qsub)  present -> $(qsub --version 2>&1 | head -1)"  || echo "no qsub"
command -v sbatch >/dev/null && echo "SLURM (sbatch) present -> $(sbatch --version 2>&1 | head -1)" || echo "no sbatch"

echo; echo "############ PBS QUEUES (if PBS) ############"
command -v qstat >/dev/null && qstat -Q 2>/dev/null | head -40

echo; echo "############ GPU NODES / TYPES ############"
if command -v pbsnodes >/dev/null; then
  echo "-- PBS pbsnodes (gpu resources) --"
  pbsnodes -a 2>/dev/null \
    | grep -iE "resources_available.ngpus|resources_available.gpu|gpu_model|resources_available.mem|resources_available.host" \
    | sort | uniq -c | sort -rn | head -40
fi
if command -v sinfo >/dev/null; then
  echo "-- SLURM sinfo (partition/gres) --"
  sinfo -o "%P %G %N %m %l" 2>/dev/null | head -40
fi

echo; echo "############ MODULES: python / cuda / vllm / gcc ############"
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | tr ' \t' '\n\n' | grep -iE "^(python|cuda|cudnn|vllm|gcc|nccl)" | sort -u | head -80
else
  echo "no 'module' command"
fi

echo; echo "############ vLLM ALREADY AVAILABLE? ############"
command -v vllm >/dev/null && { echo "vllm on PATH:"; vllm --version 2>&1 | head -1; } || echo "no vllm on PATH"
python3 -c "import vllm; print('python import vllm', vllm.__version__)" 2>&1 | head -1

echo; echo "############ PYTHON ############"
python3 --version 2>&1; command -v python3

echo; echo "############ INTERNET FROM LOGIN NODE ############"
curl -sI --max-time 8 https://huggingface.co 2>&1 | head -1 || echo "curl blocked / absent"

echo; echo "############ HF TOKEN / CACHE ############"
echo "HF_HOME=${HF_HOME:-unset}  HF_TOKEN=${HF_TOKEN:+set}  HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:+set}"
[ -d "${HF_HOME:-$HOME/.cache/huggingface}" ] && du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null

echo; echo "############ SCRATCH / QUOTA ############"
for d in "/srv/scratch/$USER" "/scratch/$USER" "$HOME"; do
  [ -d "$d" ] && { printf "%-28s " "$d"; df -h "$d" 2>/dev/null | tail -1; }
done

echo; echo "############ LLAMA-3.3-70B WEIGHTS ALREADY STAGED? ############"
find "/srv/scratch/$USER" "$HOME" -maxdepth 4 -iname "*Llama-3.3-70B*" 2>/dev/null | head

echo; echo "############ PROJECT REPO ON KATANA? ############"
find "$HOME" "/srv/scratch/$USER" -maxdepth 4 -type d -iname "COMP4952" 2>/dev/null | head

echo; echo "############ DONE ############"
