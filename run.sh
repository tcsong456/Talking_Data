bash noeve_run.sh
python auxiliary/generate_eve_preds.py --mode eve_probe --batch_size 512 --epochs 3
python main.py --optimize_result --save_path preds/eve_preds.npy
python submit.py --mode eve_probe_submit
python auxiliary/generate_eve_preds.py --mode eve_submit --batch_size 256 --epochs 3
python main.py --optimize_result --save_path preds/eve_preds.npy
python auxiliary/submit.py --mode final_submit
