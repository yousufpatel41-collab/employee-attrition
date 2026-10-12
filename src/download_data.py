"""Download the IBM sample and verify the pinned CSV checksum."""
import hashlib
from urllib.request import urlopen
from train import DATA_PATH

EXPECTED_SHA="a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"
URL="https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
if __name__=='__main__':
    raw=DATA_PATH.read_bytes() if DATA_PATH.exists() else urlopen(URL,timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA:
        raise SystemExit('Dataset differs from the pinned sample. Review the source before replacing it.')
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(raw)
    print('Verified IBM sample CSV.')
