import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(  # nbatches*batch_size = tripleTotal(train triple size)
	in_path = "./benchmarks/GADM9/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,  # ?
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/GADM9/", "link")  # no neg

# define the model
# import ipdb; ipdb.set_trace()
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100,
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
print(transe)
trainer = Trainer(model = model, data_loader = train_dataloader,
									train_times = 1, alpha = 1.0, use_gpu = False,
									checkpoint_dir='./checkpoint', save_steps=1)
# trainer.run()
# transe.save_checkpoint('./checkpoint/transe_GADM9.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_GADM9.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction(type_constrain = False)