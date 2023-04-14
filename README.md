
# Minimal setup to run Dolly 2 LLM model with 8-bit quantization

I was able to run this with an NVIDIA RTX 3080 (Laptop) with 16GB of RAM with some fiddling.
My system shows this using around ~13GB of VRAM. (`nvidia-smi` shows `13368MiB / 16384MiB` used.)

This repo loads the [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b) model using the
`transformers` library. The code in `main.py` loads it in 8-bit quantized mode.

## Setup Python Environment

```shell
python -m pip install virtualenv
virtualenv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## Install Dependencies

```shell
pip install transformers torch accelerate bitsandbytes
```

## Run the Program

```shell
python main.py
```

Output should be:
```shell
$ python main.py
<Tons of Random Warnings>
time to generate:  36.570608615875244
Who was the first President of the United States? George Washington
 Washington was the first President of the United States, serving from 1789 until his death in 1797. The first President was born in 1731 in Westmoreland County, Virginia, and died in 1799 in Mount Vernon, Virginia. His formal education ended at age 13, when he started working with his father. Washington became famous for fighting against the British in the American Revolution. He was a prominent man in the new government,
```
