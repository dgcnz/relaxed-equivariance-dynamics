echo "Finding checkpoints for wandb run id: $1"
RUNPATHS=$(find logs/train/runs -type d -name $1)
RUNPATHS=$(echo $RUNPATHS | tr ' ' '\n' | sort)
echo "Found run paths: $RUNPATHS"
# sort
for RUNPATH in $RUNPATHS
do
    echo "Run path: $RUNPATH"
    echo "Checkpoints:"
    ls $RUNPATH/checkpoints
    readlink -f $RUNPATH/checkpoints/last.ckpt
done