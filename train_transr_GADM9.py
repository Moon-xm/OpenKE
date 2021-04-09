import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/GADM9/", 
	nbatches = 150,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/GADM9/",
	sampling_mode = 'link')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 100,
	dim_r = 100,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size()
)
print(transr)
# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = True)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters("./result/GADM9_transr_transe.json")   

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 10, alpha = 1, use_gpu = True)
print('lambda: {}, gamma: {}, k: {}'.format(trainer.alpha, model_r.loss.margin.item(), transr.dim_e))
trainer.run()
transr.save_checkpoint('./checkpoint/GADM9_transr.ckpt')

# test the model
transr.load_checkpoint('./checkpoint/GADM9_transr.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)