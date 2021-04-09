import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os

# config
learning_rate = 0.001
margin = 10
dim = 100
p_norm = 1
epoch = 300
ckpt_save_steps=100
data_path = 'GeoDBpedia21'
use_gpu = True
nbatches = 100  # nbatches*batch_size = train_triple_num  只需设置一个参数即可
# batch_size = 128
in_path = './benchmarks/'+data_path+'/'
checkpoint_save_dir = './checkpoint/tanse_'+data_path+'.ckpt'
test_only = True

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = in_path, 
	nbatches = nbatches,
    # batch_size = batch_size,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(in_path, "link")

# define the model
# import ipdb; ipdb.set_trace()
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = dim, 
	p_norm = p_norm, 
	norm_flag = True)
if os.path.exists(checkpoint_save_dir):  # 加载断点继续训练
    transe.load_checkpoint(checkpoint_save_dir)
    transe.train()
    print('loading exists model sucessful...')

# print(transe)
# define the loss function
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = margin),
    batch_size = train_dataloader.get_batch_size()
)

# train the model
print(transe)
# import pdb; pdb.set_trace()
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = epoch, alpha = learning_rate, use_gpu = use_gpu, save_steps=ckpt_save_steps,checkpoint_dir=checkpoint_save_dir)
print('lamda: {}, gamma: {}, k: {}'.format(learning_rate, margin, dim))
trainer.run()
transe.save_checkpoint(checkpoint_save_dir)

# test the model
transe.load_checkpoint(checkpoint_save_dir)
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = use_gpu)
tester.run_link_prediction(type_constrain = False)