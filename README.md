# MulCode
To evaluate the compressed LM, run

python src/evalauate_ptb_lm.py --data $path_to_ptb --large True/False --W1 $path_to_W1 --W2 $path_to_W2(softmax)

To train the compression model on ptb-data, one can run src/compress.py.
For example, to train a model to compress the input embedding of ptb-large, you can
python ./src/compress.py --data ./data/ --w1 True --lm ./data/Large_LSTM.npy --rate 0.05 --M 52 --N 8 --K 12 --group_size 4000
