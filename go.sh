#echo "${SCRIPT_PATHS[@]}"
#SCRIPT_PATHS=$1
SCRIPT_PATHS=("$@")
#echo "Array content: $1"

#echo "Array content: ${SCRIPT_PATHS[@]}"
for script in "${SCRIPT_PATHS[@]}"; do

       source ~/miniconda3/bin/activate base;
#    echo " $script"
    pkill -9 -f python
    sudo rm /tmp/libtpu_lockfile
    git checkout 1a8fd8610f394248995f71bc1ac875277723a5cd
#    source ~/miniconda3/bin/activate base;
    python -u main_adv.py --yaml-path $script
#    python -u main.py --yaml-path $script
#    python -u test.py --yaml-path $script

done
