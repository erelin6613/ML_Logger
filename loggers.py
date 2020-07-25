import os
import csv
import time
import threading
import functools
import logging
import json
from datetime import datetime

def deduce_model(model):
	from torch.nn.modules.module import Module
	from sklearn.base import (ClassifierMixin,
		RegressorMixin, BaseEstimator, ClusterMixin)
	if isinstance(model, Module):
		return model.state_dict
	if issubclass(model.__class__, ClassifierMixin):
		return model.get_params


class BaseLogger:

	def __init__(self, log_dir):

		self.log_dir = log_dir

	def log(self, params, logfile='logfile.log'):
		with open(logfile, 'a') as f:
			f.write(str(time.time())+': '+str(params)+'\n')

	def print(self, logfile='logfile.log'):
		with open(logfile, 'r') as f:
			print(f.read())


class ModelLogger(BaseLogger):

	def __init__(self, model, log_dir='model_log'):

		super(ModelLogger, self).__init__(log_dir)
		self.model = model
		self.params = self.deduce_model()
		self.logfile = f'{str(self.model)}.log'
		threading.Thread(target=self.monitor).start()


	def monitor(self, timelaps=180):
		ref = hash(self.model)
		self.log(self.model) #frozenset({self.model.__getstate__()}.items))
		while True:
			# time.sleep(timelaps)
			checksum = hash(self.model)
			if ref != checksum:
				self.log({self.model})
				print('model object was changes')

def log_model(path=os.path.join('logs', 'model_log.log')):
	def log_state(model):

		@functools.wraps(model)
		def wrapper(*args, **kwargs):
			state_func = deduce_model(model())
			# print(str(state_func()))
			logger = logging.getLogger('LogModel')
			logger.setLevel(logging.INFO)
			file_handler = logging.FileHandler(path)
			log_format = '{%(levelname)s %(asctime)s %(message)s}'
			logger.addHandler(file_handler)
			logger.info(str(datetime.now().strftime(
				'%Y-%m-%d %H:%M:%S'))+'\n'+str(state_func())+'\n')
			return state_func

		return wrapper
	return log_state

def log_params(path=os.path.join('logs', 'params_log.log')):
	def log_p(params):

		@functools.wraps(params)
		def wrapper(*args, **kwargs):
			p = params.__defaults__
			logger = logging.getLogger('LogParams')
			logger.setLevel(logging.INFO)
			file_handler = logging.FileHandler(path)
			log_format = '{%(levelname)s %(asctime)s %(message)s}'
			logger.addHandler(file_handler)
			logger.info(str(datetime.now().strftime(
				'%Y-%m-%d %H:%M:%S'))+'\n'+str(p)+'\n')
			# return state_func

		return wrapper
	return log_p