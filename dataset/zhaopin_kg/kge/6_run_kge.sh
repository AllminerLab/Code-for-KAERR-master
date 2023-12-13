DGLBACKEND=pytorch dglke_train --model_name TransR --dataset zhaopin --data_path train_data/ --data_files train.txt valid.txt test.txt --format raw_udd_hrt --batch_size 1000 \
--neg_sample_size 200 --hidden_dim 16 --gamma 19.9 --lr 0.25 --max_step 25000 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --test --gpu 0 --mix_cpu_gpu --save_path ./out