# zcomp

Zero-shot composed retrieval

### Demo inference command 
python src/demo.py --openai-pretrained --resume ./pic2word_model.pt --retrieval-data imgnet --query_file "./data/imgnet/real/n01770393/ILSVRC2012_val_00003023.JPEG,./data/imgnet/real/n01770393/ILSVRC2012_val_00031684.JPEG" --prompts "a cartoon of *" --demo-out ./demo_result --gpu 1 --model ViT-L/14

### Evaluation for metrics command
python src/eval_retrieval.py --openai-pretrained --resume ./pic2word_model.pt --eval-mode cirr --gpu 0 --model ViT-L/14 --distributed --dist-url tcp://127.0.0.1:6101

### Training command

#### For raw folders
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u src/main.py --save-frequency 1 --train-data="/home/yucheng/cc3m_byte/image_byte_224" --dataset-type directory --warmup 10000 --batch-size=128  --lr=1e-4 --wd=0.1  --epochs=30 --workers=6 --openai-pretrained --model ViT-L/14  --dist-url tcp://127.0.0.1:6102 --seed 999

#### For webdataset format
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1,2 python  -u src/main.py --save-frequency 1 --train-data="/mount/ccai_nas/yunhong/jiaheng/CC3M/webdataset_224/cc3m-{00000..00255}.tar" --dataset-type webdataset --warmup 10000 --batch-size=128  --lr=1e-4 --wd=0.1  --epochs=30 --workers=2 --openai-pretrained --model ViT-L/14  --dist-url tcp://127.0.0.1:6102 --dist-backend gloo 
