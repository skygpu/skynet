# skynet
### decentralized compute platform

To launch a worker:
```
# create and edit config from template
cp skynet.ini.example skynet.ini

# create python virtual envoirment 3.10+
python3 -m venv venv

# enable envoirment
source venv/bin/activate

# install requirements
pip install -r requirements.txt
pip install -r requirements.cuda.0.txt
pip install -r requirements.cuda.1.txt
pip install -r requirements.cuda.2.txt

# install skynet
pip install -e .

# test you can run this command
skynet --help

# to launch worker
skynet run dgpu

```
