# assumes shell/ACSIncome-CA-2014/run_tree.sh was run before this
python3 -m src.scripts.eval_embeddings -D -f result/ACSIncome-CA-2014/tree -c config/eval_config.py -r result/_eval/ACSIncome-CA-2014 -m tree --force --key 0%3
python3 -m src.scripts.eval_embeddings -D -f result/ACSIncome-CA-2014/tree -c config/eval_config.py -r result/_eval/ACSIncome-CA-2014 -m tree --force --key 1%3
python3 -m src.scripts.eval_embeddings -D -f result/ACSIncome-CA-2014/tree -c config/eval_config.py -r result/_eval/ACSIncome-CA-2014 -m tree --force --key 2%3
python3 -m src.scripts.merge_embeddings --dataset ACSIncome-CA-2014 --name tree --mod 3
