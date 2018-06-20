#!/bin/bash
IN=$1
JOB=${IN%.*}
BASENAME=$(basename $JOB)
PWD=`pwd`

cat > $JOB.sh <<!EOF
#!/bin/bash
#SBATCH -p kemi1
#SBATCH --account=chemistry
#SBATCH -c 1
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --error=$HOME/logs/$BASENAME\_%j.err
#SBATCH --output=$HOME/logs/$BASENAME\_%j.out
#SBATCH --mem=4gb
#SBATCH --time=2-0

osprey worker -j1 -n 50 $IN > $JOB\.out

!EOF

sbatch $JOB.sh
