# Athena
A deeplearning framework for 10 billlion parameters DL model.

At present, Tenbillionflow supports for DNN training for both CPU and GPU with manually operators on numpy and CUDA C++ implemantation.

## Installation
1. Clone the respository.
2. Edit the athena.exp file and set the environment path for python.

```bash
source athena.exp
```

3. Compile each system by ```make all```
For standalone development:
```bash
make clean
make athena version=cpu -j 32
make athena version=gpu -j 32
```

4. Install graphviz to support graph board visualization

```bash
sudo apt-get install graphviz
sudo pip install graphviz
```

5. Install pynvml to support vdnn scheduler policy.

```bash
sudo pip install nvidia-ml-py
```
## For mkl support

```bash
make clean
make athena version=mkl -j 32
cd tests
python mnist_mlp_no_bias.py
```

To compare with numpy, rebuild with cpu version and run the same test script.

