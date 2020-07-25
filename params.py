from dataclasses import dataclass, replace, field
from typing import List
from Recognition.Config import HP
@dataclass
class E2E_Config():
	#Recognition Parameters
	recog_model_path: str
	lang: str
	recog_hp: HP
	#Detection Parameters
	det_model_path: str
	arch: str = 'resnet50'
	binary_th: float = 1.0
	kernel_num: int = 7
	scale: int = 1
	long_size: int = 2240
	min_kernel_area: int = 5.0
	min_area: float = 800.0
	min_score: float = 0.93
	

recog_hp_config = HP(experiment_name='-',exp_dir='-')

e2e_config = E2E_Config(recog_model_path='/home/azhar/hin_best.pth',det_model_path='/home/azhar/PSEnet_best.pth.tar.part',lang='hin',recog_hp=recog_hp_config)

