from urllib.request import urlopen
from train import DATA_PATH

if __name__=="__main__":
    DATA_PATH.parent.mkdir(exist_ok=True)
    if DATA_PATH.exists(): raise SystemExit("Dataset already exists")
    url="https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    with urlopen(url,timeout=60) as response: DATA_PATH.write_bytes(response.read())
    print(DATA_PATH)
