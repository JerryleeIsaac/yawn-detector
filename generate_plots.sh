metrics=("f1" "accuracy" "precision" "recall")
experiments="exp/resnet50-pretrained exp/mobilenetv3-pretrained exp/mobilenetv3 exp/resnet50 exp/resnet101-pretrained"


for metric in "${metrics[@]}"; do
	dvc plots diff $experiments -t smooth --targets pipeline_outs/results/train-plots.json  -y $metric --x-label "Epoch" --title "Train $metric Score over Epoch" --open
	dvc plots diff $experiments -t smooth --targets pipeline_outs/results/test-plots.json  -y $metric --x-label "Epoch" --title "Test $metric Score over Epoch" --open
done
dvc plots diff $experiments -t smooth --targets pipeline_outs/results/losses.json  --title "Validation loss over training steps" --open