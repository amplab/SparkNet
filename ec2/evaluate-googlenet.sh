NET=/root/caffe/models/bvlc_googlenet/train_val.prototxt
PREFIX=/imagenet2/1gpu/bvlc_googlenet_quick_iter_
LOGFILE=/imagenet2/1gpu/acc.txt

for i in `seq 0 4000 300000`; do
	python evaluate.py 0 $NET $PREFIX $LOGFILE;
done
