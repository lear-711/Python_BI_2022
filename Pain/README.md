This file gives you the instruction, how you can run file pain.py. Together we will create file requirements.txt, where all necessary packages and their versions will be writed.

After this commands you can easily reproduce Michael's results by running scripts locally.

Good luck!

### **OS**: 
macOS Mojave version 10.14.6
### **Python version**: 
python3.10


# Instruction:

1. **First of all we have to create virtual environment:**

```
python -m venv environment

cd environment/

source environment/bin/activate
```

2. **Clone fork from Michael's repository:**

```
git clone git@github.com:lear-711/Virtual_environment_research.git

cd Virtual_environment_research/
```

Here we have file pain.py!

3. **Install necessary packages:**

```
pip install google

pip install math

pip install typing

pip install kivy

pip install bs4

pip install requests

pip install Bio

pip install aiohttp

pip install pandas

pip install scipy

pip install scanpy

pip install opencv-python
```

4. **Module "match" requires version of python 3.10:**

```
conda update --all

conda install python==3.10
```

5. **Then it turned out that some packages installations work differently on python 3.10. So I reinstalled some packages:**

google package's name is different:
```
pip install google-api-python-client
```


```
python3.10 -m pip install kivy

python3.10 -m pip install Bio

python3.10 -m pip install aiohttp

python3.10 -m pip install pandas

python3.10 -m pip install scipy

python3.10 -m pip install scanpy

python3.10 -m pip install opencv-python
```
But with this package there were veeryyy maaanyyyy problems, so I decidid to do next:

```
conda install -c conda-forge opencv

python3.10 -m pip install numpy

python3.10 -m pip install lxml
```

6. **Run the file:**

```
python3.10 pain.py
```

7. **Print list of nesserary packages and their versions:**

```
pip list --format=freeze
```

8. **Write all required packages with their versions to file requirements.txt:**

```
pip list --format=freeze > requirements.txt
```

## My congratulations! Now you can run pain.py! 
## Feel it!