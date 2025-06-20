A Python chess engine using deep neural network evaluation and a quiescent variant of the Predictor + UCB applied to trees algorithm (PUCT).\
Estimated ELO ~1600 but aspires to be the strongest Python chess engine!\
Please contact me on Discord (@yoman417) or email walterliu417@gmail.com if there are any UCI bugs.\
A list of UCI options can be found in uci_options.md in the docs folder.\
For technical documentation on the engine, see technical.md in the docs folder.

**Usage**\
In the Colab Notebooks folder, use parakeet_implementation.ipynb to run as a Lichess bot, and parakeet_tuning.ipynb to run games on fastchess.
Parakeet uses the UCI protocol.

**Building from source**\
On any terminal:
Acquire the source code.
```
git clone https://github.com/walterliu417/parakeet.git
cd parakeet
python -m venv venv
```
Activate the virtual Python environment.\
Linux:
```
source venv/bin/activate
```
Windows:
```
venv/Scripts/activate.bat
```
Install required libraries:
```
pip install numpy chess
```
On a GPU-enabled environment:
```
pip install onnxruntime-gpu
```
Otherwise:
```
pip install onnxruntime
```
Run the engine:
```
python parakeet.py
```
If you wish to obtain an executable, simply use
```
pip install pyinstaller
pyinstaller --onefile parakeet.py
```

**Future plans**
- Adaptive time management
- Gradient boosting networks (predicting the error between the current network and the target)
- More hyperparameter tuning instead of using random search
- C++ implementation for best performance.

**Credits**
- AlphaZero and Leela Chess for inspiring this project.
- Engine Programming Discord for all the help the friendly people there gave me.
- SOAP: Improving and Stabilising Shampoo using Adam. https://arxiv.org/abs/2409.11321
- Lichess Evaluation and Puzzle Databases.
- Fastchess, for testing the engine and tuning hyperparameters.
- ChatGPT, for basically giving me a crash course on supervised learning and acting as a platform to bounce ideas off of.
