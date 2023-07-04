#!/bin/bash
# Path: scripts/helper/l2v/run_l2v_distances.sh
# This script is used to get l2v distances between languages for a given dataset.


datasets=( "xnli" "udpos" "PAN-X" "tydiqa" )

for dataset in "${datasets[@]}"
do
    if [ ${dataset} == "xnli" ]
    then
        languages=ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh
    elif [ ${dataset} == "udpos" ]
    then
        languages=Arabic,Basque,Bulgarian,Chinese,Dutch,English,Estonian,Finnish,French,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Kazakh,Korean,Malay,Marathi,Persian,Portuguese,Russian,Spanish,Swahili,Swedish,Tamil,Telugu,Thai,Turkish,Urdu,Vietnamese,Yoruba
    elif [ ${dataset} == "PAN-X" ]
    then
        languages=af,ar,bg,bn,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,nl,pt,ru,sw,ta,te,th,tl,tr,ur,vi,yo,zh
    elif [ ${dataset} == "tydiqa" ]
    then
        languages=ar,bn,en,fi,id,ko,ru,sw,te
    fi
    echo "Getting distances for ${dataset}"
    python src/helper/l2v/get_distances.py \
    --languages ${languages} \
    --dataset_name ${dataset} \
    --output_path src/helper/l2v \
    --l2v_code_map_path src/helper/l2v/code_map.json
done
