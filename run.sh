for m in "vanilla_bert" "cedr_pacrr" "cedr_knrm" "cedr_drmm"
do
	python train.py --model $m
done
