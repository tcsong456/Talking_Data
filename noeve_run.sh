python auxiliary/generate_stack_preds.py --mode eve_submit --batch_size 256 --epochs 3

root="preds/submission"
for mode in 'val' 'test';do
  root_dir="$root/$mode/"
  ls_files=$(ls $root_dir)
  for folder in "nn_lr" "lgb";do
    makedir_folder="$root_dir/$folder"
    mkdir $makedir_folder
    if grep -q "val"<<<$mode;then
      cp "$root_dir/label.npy" "$root_dir/$folder"
    fi
  done
  for file in $ls_files;do
    cur_file="$root_dir/$file"
    if (grep -q "nn"<<<$file || grep -q "lr"<<<$file) && [[ -f $cur_file ]];then
      mv $cur_file "$root_dir/nn_lr"
    fi
    if grep -q "lgb"<<<$file && [[ -f $cur_file ]];then
      mv $cur_file "$root_dir/lgb"
    fi
  done
done

python main.py --optimize_result --save_path preds/submission/no_eve_nnlr.npy \
--pred_store_path preds/submission/val/nn_lr

python main.py --optimize_result --save_path preds/submission/no_eve_lgb.npy \
--pred_store_path preds/submission/val/lgb

python auxiliary/consolidate_noeve.py
