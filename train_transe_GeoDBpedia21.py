import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/GeoDBpedia21/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/GeoDBpedia21/", "link")

# define the model
# import ipdb; ipdb.set_trace()
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True)

# print(transe)
# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 10),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
print(transe)
# import pdb; pdb.set_trace()
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 10000, alpha = 0.1, use_gpu = True)
print('lamda: {}, gamma: {}, k: {}'.format(trainer.alpha, model.loss.margin.item(), transe.dim))
trainer.run()
transe.save_checkpoint('./checkpoint/transe_GADM9.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_GADM9.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)