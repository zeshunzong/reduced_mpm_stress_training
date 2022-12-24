from pytorch_lightning.utilities.rank_zero import rank_zero_only
from util import *
from CROMnet_secondary import *
from SimulationDataset import *
from torch.utils.data import DataLoader


class Exporter(object):
    def __init__(self, weight_path):
        self.weight_path = weight_path
       
    @rank_zero_only
    def export(self, ):

        net = CROMnet_secondary.load_from_checkpoint(self.weight_path)
        device = findEmptyCudaDevice()

        net_dec = net.decoder.to(device)
        
        data_list = DataList(net.data_format['data_path'], 1.0)
        dataset = SimulationDataset(net.data_format['data_path'], data_list.data_list)
        trainloader = DataLoader(dataset, batch_size=1)
        data_batched = next(iter(trainloader))

        encoder_input = data_batched['encoder_input'].to(device) # shape is [batch_size, nparticles, lbl+input_dim]
        batch_size_local = encoder_input.size(0)
        x = encoder_input.view(batch_size_local * encoder_input.size(1), encoder_input.size(2)) #essentially (label, X), shape = [batch_size*n_particle, lbl + input_dim]

        x_original = x

        x = x.detach()
        x.requires_grad_(True)
        q = net_dec(x)
        q = q.view(batch_size_local, -1, q.size(1)).detach()
        
        net_dec_jit = net_dec.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        
        q_jit = net_dec_jit.forward(x)
        q_jit = q_jit.view(batch_size_local, -1, q_jit.size(1))

        assert(torch.norm(q-q_jit)<1e-10)

        #print("decoder trace finished")

        encoder_input = data_batched['encoder_input'].to(device)
        output_regular, _, _ = net.forward(encoder_input)

        assert(torch.norm(output_regular-q_jit)<1e-10)

        #print("full network trace finished")
        dec_jit_path = os.path.splitext(self.weight_path)[0]+"_dec.pt"

        print('decoder torchscript path: ', dec_jit_path)
        torch.jit.save(net_dec_jit, dec_jit_path)  
        #net_dec_jit.save(dec_jit_path)

        # trace grad
        x = x_original
        num_sample = 10
        x = x[0:num_sample, :]
        
        net_dec_func_grad = NetDecFuncGrad(net_dec)
        net_dec_func_grad.to(device)

        grad, y = net_dec_func_grad(x)
        grad = grad.clone() # output above comes from inference mode, so we need to clone it to a regular tensor
        y = y.clone()
        
        grad_gt, y_gt = net_dec.computeJacobianFullAnalytical(x)

        encoder_input.requires_grad = True
        encoder_input.retain_grad()

        outputs_local, _, decoder_input = net.forward(encoder_input)
        
        grad_gt_auto = computeJacobian(decoder_input, outputs_local)
        grad_gt_auto = grad_gt_auto.view(grad_gt_auto.size(0)*grad_gt_auto.size(1), grad_gt_auto.size(2), grad_gt_auto.size(3))
        grad_gt_auto = grad_gt_auto[0:num_sample, :, :]

        criterion = nn.MSELoss()
        assert(criterion(grad_gt_auto, grad_gt)<1e-2)
        assert(criterion(grad, grad_gt)<1e-2)
        assert(criterion(y, y_gt)<1e-2)
        
        # grad, y = net_auto_dec_func_grad(x)
        with torch.jit.optimized_execution(True):
            net_dec_func_grad_jit = net_dec_func_grad.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
            grad_jit, y_jit = net_dec_func_grad_jit(x)
        
        assert(torch.norm(grad-grad_jit)<1e-2)
        assert(torch.norm(y-y_jit)<1e-2)

        #print("decoder gradient trace finished")

        dec_func_grad_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_func_grad.pt"
        print('decoder gradient torchscript path: ', dec_func_grad_jit_path)
        net_dec_func_grad_jit.save(dec_func_grad_jit_path)

        net_dec_func_grad.cpu()
        net_dec_func_grad_jit = net_dec_func_grad.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        dec_func_grad_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_func_grad_cpu.pt"
        #print('decoder gradient torchscript path (cpu): ', dec_func_grad_jit_path)
        net_dec_func_grad_jit.save(dec_func_grad_jit_path)


        # net_enc_jit_load = torch.jit.load(enc_jit_path)
        net_dec_jit_load = torch.jit.load(dec_jit_path)

        encoder_input = data_batched['encoder_input'].to(device)
       
        x = x_original
        q_jit_load = net_dec_jit_load.forward(x)
        q_jit_load = q_jit_load.view(batch_size_local, -1, q_jit_load.size(1))
        assert(torch.norm(q_jit_load-q_jit)<1e-10)


        net_dec.cpu()
        x = x.cpu()
        net_dec_jit = net_dec.to_torchscript(method = 'trace', example_inputs = x, check_trace=True, check_tolerance=1e-20)
        dec_jit_path = os.path.splitext(self.weight_path)[0]+"_dec_cpu.pt"
        print('decoder torchscript path (cpu): ', dec_jit_path)
        net_dec_jit.save(dec_jit_path)