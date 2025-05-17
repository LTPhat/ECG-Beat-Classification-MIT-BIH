from data.process import ecg_processdata 
# from data.process import ppg_processdata 
# from data.process import har_processdata 
import os
import sys
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)
print("Downloading and processing all datasets")
ecg_processdata.main()
# ppg_processdata.main()
# har_processdata.main()