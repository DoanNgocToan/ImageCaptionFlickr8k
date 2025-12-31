# This is not a benchmark (for real)
This repo use ResNet50/ EfficientNet B0, B2, B3, B4 as Encoder; LSTM (with data injection) / LSTM (with data injection) + Bahdanau Attention / Transformer (CE Loss only) as Decoder; evaluate by BLEU (Exp) and METEOR as default, BLEU (linear) is for further research
# How to run any files (Window)
1. Open cmd in repo folder
2. Enable env if you have to
3. Intstall dependencies
   pip install -r "requirements.txt"
5. Ensure you have your CUDA compatible with pytorch and GPU driver for best performance
6. Run the following prompt:

python -u "{file_name.py}" --mode={mode} (for ResNet50 file) or

python -u "{file_name.py}" --variant={variant} --mode={mode} (for EfficientNet file)

You need to properly write your file name and arguments (check source code for supported arguments and their appropriate values)
# How to run flickr30k
Just adjust CONFIG, FILE_PATHS and voilà
