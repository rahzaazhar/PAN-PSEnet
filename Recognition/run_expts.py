import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--config_list',nargs='+',help='enter names of Configs')
args = parser.parse_args()
config_list = args.config_list

for config in config_list:
	#config = "--"
	subprocess.call(["python","trainv2.py","--config_name",config])

