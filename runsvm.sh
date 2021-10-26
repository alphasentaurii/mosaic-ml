#!/bin/bash -xu

export SVM_QUALITY_TESTING=on

# DATASETS=('ib3m53' 'icon04' 'ic2e01' 'ibcf61' 'idnco5' 'icxoy4' 'ibja06' 'icas0j' 'ic0556' 'idr0b3' 'ib2p57' 'id5d10' 'j8rq33' 'j8mq50' 'j8ep06' 'j90d06' 'j8dnf1' 'j8qo05' 'j8za03' 'jb3716' 'j8rh03' 'j97k03' 'j8mbnl' 'j97e06' 'jb4g02' 'j9zp15' 'j8y503' 'ja2h01' 'j9x117' 'ja5m01')
#DATASETS=('ib3m53' 'icon04' 'ic2e01' 'ibcf61' 'idnco5' 'icxoy4' 'ibja06' 'icas0j' 'ic0556' 'idr0b3' 'id5d10')
DATASETS=('j8rq33' 'j8mq50' 'j8ep06' 'j90d06' 'j8dnf1' 'j8qo05' 'j8za03' 'jb3716' 'j8rh03' 'j97k03' 'j8mbnl' 'j97e06' 'jb4g02' 'j9zp15' 'j8y503' 'ja2h01' 'j9x117' 'ja5m01')

for dataset in "${DATASETS[@]}"
do
    #python corrupt_svm.py ${dataset} mfi
    #python corrupt_svm.py ${dataset} mfi -m=stat
    python corrupt_svm.py ${dataset} mfi -e=sub
    # python corrupt_svm.py ${dataset} mfi -e=sub -m=stat
    visits=(`find ${dataset}_* -maxdepth 0`)
    for v in "${visits[@]}"
    do
        cd ${v}; input_file=`find . -name "*input.out"`
        runsinglehap "${input_file}"
        cd ../
    done
done