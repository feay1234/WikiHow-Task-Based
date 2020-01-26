for m in "vanilla_bert" "cedr_pacrr" "cedr_knrm" "cedr_drmm"
#for m in "vanilla_bert"
do
	#for d in "query" "query-wiki" "query-wiki-question" "query-wiki-ms" "query-wiki-question-ms"
	for d in "query-wiki-ms-nostopword" "query-wiki-question-ms-nostopword"
	#for d in "query-wiki-question-ms"
	do
		#python train5foldCV.py --model $m --data $d --datafiles data/cedr/$d.tsv --fold 1 --epoch 1
		python train5foldCV.py --model $m --data $d --datafiles data/cedr/$d.tsv --epoch 10
	done
done
