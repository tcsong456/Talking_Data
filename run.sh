#bash noeve_run.sh
#python auxiliary/transform_topic_feature.py --num_of_cores 6
#python auxiliary/generate_eve_preds.py --mode eve_probe --batch_size 256 --epochs 3
python main.py --optimize_result --save_path preds/eve_preds.npy
python submit.py --mode eve_probe_submit
python auxiliary/generate_eve_preds.py --mode eve_submit --batch_size 256 --epochs 3
python main.py --optimize_result --save_path preds/eve_preds.npy
python submit.py --mode final_submit
