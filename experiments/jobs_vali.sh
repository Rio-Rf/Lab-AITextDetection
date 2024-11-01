# Experiments for the LD News datasets with generations from gpt-3.5-turbo model
# run_vali.pyを引数を指定して実行するプログラム

python run_vali.py \
  --dataset_path ../datasets/JP/oscar/oscar-vali.jsonl \
  --dataset_name OSCAR \
  --human_sample_key text \
  --machine_sample_key gpt-3.5-turbo_generated_text_wo_prompt \
  --machine_text_source gpt-3.5-turbo