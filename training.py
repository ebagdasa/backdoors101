from datetime import datetime
import argparse

# from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
# import torchvision.models as models

# from models.resnet import resnet18, resnet50
import yaml
import shutil
# from models.resnet import *
# from models.word_model import RNNModel
from tasks.batch import Batch
from utils.helper import Helper
from utils.utils import *
# from utils.image_helper import ImageHelper
from prompt_toolkit import prompt
# from utils.min_norm_solvers import *
logger = logging.getLogger('logger')
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
from scipy import stats


def compute_loss(helper, model, data, criterion):



    return

def train_new(helper: Helper):
    model = helper.task.model
    criterion = helper.task.criterion

    for data in helper.task.train_loader:
        batch = helper.task.get_batch(data)

        loss = compute_loss(helper, model, batch, criterion)

        loss.backward()

        helper.task.optimizer.step()

    return




# @profile
def train(run_helper, model, optimizer, criterion, epoch):
    train_loader = run_helper.train_loader
    if run_helper.backdoor and run_helper.data != 'nlp' and run_helper.switch_to_eval:
        model.eval()
        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        if run_helper.fixed_model:
            run_helper.fixed_model.eval()
    else:
        model.train()
    fixed_model = run_helper.fixed_model

    if run_helper.gan:
        run_helper.discriminator.eval()

    tasks = run_helper.losses
    running_scale = dict()
    running_losses = {'loss': 0.0}
    for t in run_helper.ALL_TASKS:
        running_losses[t] = 0.0
        running_scale[t] = 0.0

    loss = 0
    last_loss = 1000

    for i, data in enumerate(train_loader, 0):


        # if i >= 1000 and run_helper.data == 'imagenet':
        #     break
        if run_helper.slow_start:
            if i >= 1000 and run_helper.data == 'imagenet':
                run_helper.normalize = 'loss+'
            else:
                run_helper.normalize = 'eq'
        if run_helper.timing:
            torch.cuda.synchronize()
            tt = 0
        # get the inputs
        tasks = run_helper.losses
        if run_helper.data == 'multimnist':
            inputs, labels = data
            # second_labels = second_labels.to(run_helper.device)
        elif run_helper.data == 'pipa':
            inputs, labels, second_labels, _ = data
            second_labels = second_labels.to(run_helper.device)
        elif run_helper.data == 'nlp':
            inputs, labels = data.text, data.label
        else:
            inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.to(run_helper.device)
        labels = labels.to(run_helper.device)

        if not run_helper.backdoor or random.random()>run_helper.alternating_attack:
            # for m in model.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         m.train()
            t = 0
            outputs, _ = model(inputs)
            run_helper.record_time(t,'forward')

            loss = criterion(outputs, labels).mean()

            loss_data = dict()
            t = 0
            loss.backward()
            run_helper.record_time(t,'backward')

            t = 0
            optimizer.step()
            run_helper.record_time(t,'step')
        else:
            if run_helper.clip_batch:
                inputs = inputs[:run_helper.clip_batch]
                labels = labels[:run_helper.clip_batch]
            t = 0
            inputs_back, labels_back = poison_train(run_helper, inputs,
                                                    labels, run_helper.backdoor_label,
                                                    run_helper.poisoning_proportion)
            run_helper.record_time(t,'poison')
            ## don't attack always

            if 'sums' in tasks:
                inputs_sum, labels_sum = poison_pattern_mnist(inputs, labels, 8, 1.1, multi=True, sum=True)
            if run_helper.data == 'pipa':
                labels_back.copy_(second_labels)

            if run_helper.nc and not run_helper.new_nc_evasion:
                run_helper.mixed.grad_weights(mask=True, model=False)
                # inputs_back_full, labels_back_full = poison_train(run_helper, inputs,
                #                                                   labels, run_helper.backdoor_label,
                #                                                   1.1)
                tasks = ['nc', 'mask_norm']
                run_helper.mixed.zero_grad()
                scale = {'mask_norm': 0.001, 'nc': 0.999}
                nc_criterion = nn.CrossEntropyLoss(run_helper.nc_tensor_weight, reduction='none')
                loss_data, grads = run_helper.compute_losses(tasks, run_helper.mixed, nc_criterion, inputs,
                                                             inputs_back,
                                                             labels, labels_back, fixed_model, compute_grad=False)

                loss_flag = True
                for zi, t in enumerate(tasks):
                    if zi == 0:
                        loss = scale[t] * loss_data[t]
                    else:
                        loss += scale[t] * loss_data[t]
                if loss_flag:
                    t = 0
                    loss.backward()
                    run_helper.record_time(t,'backward')

                else:
                    loss = torch.tensor(0)
                t = 0
                helper.mixed_optim.step()
                run_helper.record_time(t,'step')

                run_helper.mixed.grad_weights(mask=False, model=True)
                tasks = helper.losses

            if run_helper.normalize != 'eq' and len(tasks)>1:
                loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                             labels, labels_back, fixed_model, compute_grad=True)
                if 'sums' in tasks:
                    loss_data['sums'], grads['sums'] = run_helper.compute_backdoor_loss(model, criterion,
                                                                                    inputs_sum, labels,
                                                                                    labels_sum,
                                                                                    grads=True)
                if run_helper.nc:
                    if run_helper.new_nc_evasion:
                        inputs_nc, labels_nc = poison_nc(inputs,
                                                         labels, run_helper.backdoor_label,
                                                         run_helper.poisoning_proportion)
                        loss_data['nc_adv'], grads['nc_adv'] = helper.compute_nc_loss(model, inputs_nc,
                                                                                            labels_nc,
                                                                                            grads=True)
                    else:
                        loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed,
                                                                                           criterion, inputs, labels,grads=True)

                for t in tasks:
                    if loss_data[t].mean().item() == 0.0:
                        loss_data.pop(t)
                        grads.pop(t)
                        tasks = tasks.copy()
                        tasks.remove(t)
                if len(tasks)>1:
                    t = 0
                    try:
                        scale = MinNormSolver.get_scales(grads, loss_data, run_helper.normalize, tasks, running_scale, run_helper.log_interval)
                    except TypeError:
                        print('type error. exiting')
                        break
                    run_helper.record_time(t,'scales')
                else:
                    scale = {tasks[0]: 1.0}
            else:
                if len(tasks) > 1:
                    scale = dict()
                else:
                    scale = {tasks[0]: 1.0}
                loss_data, grads = run_helper.compute_losses(tasks, model, criterion, inputs, inputs_back,
                                                             labels, labels_back, fixed_model, compute_grad=False)
                if 'sums' in tasks:
                    loss_data['sums'], grads['sums'] = run_helper.compute_backdoor_loss(model, criterion,
                                                                              inputs_sum, labels,
                                                                              labels_sum,
                                                                              grads=False)
                if helper.nc:
                    loss_data['nc_adv'], grads['nc_adv'] = helper.compute_normal_loss(run_helper.mixed, criterion, inputs,
                                                                                  labels, grads=False)
            loss_flag = True
            if helper.normalize == 'eq':
                for t in tasks:
                    scale[t] = run_helper.params['losses_scales'].get(t, 0.5)
                    running_scale[t] = scale[t]
            for t in tasks:
                run_helper.save_dict[f'loss.{t}'].append(loss_data[t].item())
                run_helper.save_dict[f'scale.{t}'].append(scale[t])
            for zi, t in enumerate(tasks):
                if zi == 0:
                    loss = scale[t] * loss_data[t]
                else:
                    loss += scale[t] * loss_data[t]
            run_helper.last_scales = scale
            if loss_flag:

                if run_helper.dp:
                    saved_var = dict()
                    for tensor_name, tensor in model.named_parameters():
                        saved_var[tensor_name] = torch.zeros_like(tensor)

                    for j in loss:
                        t = 0
                        j.backward(retain_graph=True)
                        run_helper.record_time(t,'backward')
                        torch.nn.utils.clip_grad_norm_(model.parameters(), run_helper.S)
                        for tensor_name, tensor in model.named_parameters():
                            new_grad = tensor.grad
                            saved_var[tensor_name].add_(new_grad)
                        model.zero_grad()

                    for tensor_name, tensor in model.named_parameters():
                        if run_helper.device.type == 'cuda':
                            noise = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, run_helper.sigma)
                        else:
                            noise = torch.FloatTensor(tensor.grad.shape).normal_(0, run_helper.sigma)
                        saved_var[tensor_name].add_(noise)
                        tensor.grad = saved_var[tensor_name] / run_helper.batch_size

                    loss = loss.mean()
                    for t, l in loss_data.items():
                        loss_data[t] = l.mean()
                else:
                    t = 0
                    loss.backward()
                    run_helper.record_time(t,'backward')
            else:
                loss = torch.tensor(0)
            t = 0
            optimizer.step()
            run_helper.record_time(t,'step')
        running_losses['loss'] += loss.item()/(run_helper.log_interval * min(run_helper.alternating_attack, 1))
        for t, l in loss_data.items():
            running_losses[t] += l.item()/(run_helper.log_interval * min(run_helper.alternating_attack, 1))

        if i > 0 and i % run_helper.log_interval == 0 and not run_helper.timing:
            logger.warning(f'scale: {running_scale}')
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_losses['loss']))
            run_helper.plot(epoch * len(train_loader) + i, running_losses['loss'], 'Train_Loss/Train_Loss')
            running_losses['loss'] = 0.0
            norms = {'latent': [], 'latent_fixed': []}

            for t in helper.ALL_TASKS:
                if running_losses[t] == 0.0:
                    running_scale[t] = 0
                    continue
                logger.info('[%d, %5d] %s loss: %.3f' %
                            (epoch, i + 1, t, running_losses[t]))
                run_helper.plot(epoch * len(train_loader) + i, running_losses[t], f'Train_Loss/{t}')
                run_helper.plot(epoch * len(train_loader) + i, running_scale[t], f'Train_Scale/{t}')
                running_losses[t] = 0.0
                running_scale[t] = 0

        if run_helper.timing:
            run_helper.record_time(tt,'total')


def test(run_helper, model, criterion, epoch, is_poison=False, sum=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    prec_5 = list()
    prec_1 = list()
    i = 0
    correct_labels = []
    predict_labels = []
    with torch.no_grad():
        for i, data in enumerate(run_helper.test_loader):
            # if i > 100 and run_helper.data == 'imagenet': # and is_poison:
            #     break
            if run_helper.data == 'multimnist':
                inputs, labels = data
                # inputs, labels, second_labels = data
                # second_labels = second_labels.to(run_helper.device)
            elif run_helper.data == 'pipa':
                inputs, labels, second_labels, _ = data
                if run_helper.data == 'pipa' and is_poison and (second_labels==0).sum().item()==second_labels.size(0):
                    continue
                second_labels = second_labels.to(run_helper.device)
            elif run_helper.data == 'nlp':
                inputs, labels = data.text, data.label
            else:
                inputs, labels = data
            inputs = inputs.to(run_helper.device)
            labels = labels.to(run_helper.device)
            if is_poison:
                poison_test(run_helper, inputs,
                             labels, run_helper.backdoor_label, sum)
                if run_helper.data == 'pipa':
                    labels.copy_(second_labels)
            outputs, _ = model(inputs)

            loss = criterion(outputs, labels).mean()
            total_loss += loss.item()
            if run_helper.data == 'nlp':
                predicted = torch.round(torch.sigmoid(outputs.data))
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if run_helper.data == 'imagenet':
                c1, c5 = accuracy(outputs, labels, (1, 5))
                prec_5.append(c5)
                prec_1.append(c1)
            if run_helper.data == 'pipa' and is_poison:
                total -=    (labels == 0).sum().item()
                correct -= (predicted[labels == 0] == 0).sum().item()

            # predict_labels.extend([x.item() for x in predicted])
            # correct_labels.extend([x.item() for x in labels])
    main_acc = 100 * correct / total
    if run_helper.data == 'imagenet':
        logger.warning(f'Epoch {epoch}. Correct: {correct} Poisoned: {is_poison}.'
                       f' Accuracy: Top-5 {np.mean(prec_5):.2f}%. Top-1 {np.mean(prec_1):.2f}% '
                       f'Loss: {total_loss/total:.4f}')
    else:
        logger.warning(f'Epoch {epoch}. Correct: {correct} Poisoned: {is_poison}. Accuracy: {main_acc}%. Loss: {total_loss/total:.4f}')
    if is_poison:
        run_helper.plot(x=epoch, y=main_acc, name="accuracy/poison")
    else:
        run_helper.plot(x=epoch, y=main_acc, name="accuracy/normal")

    # if helper.tb:
    #     fig, cm = plot_confusion_matrix(correct_labels, predict_labels, labels=list(range(10)), normalize=True)
    #     helper.writer.add_figure(figure=fig, global_step=0, tag=f'images/normalized_cm_{epoch}_{is_poison}')
    #     helper.writer.flush()
    return main_acc, total_loss


def run(run_helper):

    # load data
    if run_helper.data == 'cifar':
        run_helper.load_cifar10(run_helper.batch_size)
        model = ResNet18(num_classes=len(run_helper.classes))
        run_helper.fixed_model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'cifar_vgg':
        run_helper.load_cifar10(run_helper.batch_size)
        model = models.vgg19(num_classes=len(run_helper.classes))
        run_helper.fixed_model = models.vgg19(num_classes=len(run_helper.classes))
    elif run_helper.data == 'mnist':
        run_helper.load_mnist(run_helper.batch_size)
        model = Net()
    elif run_helper.data == 'multimnist':
        run_helper.load_multimnist(run_helper.batch_size)
        model = Net(len(run_helper.classes))
        # model = ResNet18(len(run_helper.classes))
        # model = ResNet18(num_classes=len(run_helper.classes))
    elif run_helper.data == 'imagenet':
        run_helper.load_imagenet()
        # model = vgg11(pretrained=True)

        model = resnet18(pretrained=run_helper.pretrained)
        run_helper.fixed_model = model #resnet18(pretrained=True)
        run_helper.fixed_model.to(run_helper.device)
    elif run_helper.data == 'pipa':
        run_helper.load_pipa()
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512 , 5)
        run_helper.fixed_model = model
    elif run_helper.data == 'celeba':
        run_helper.load_celeba()
        model = resnet50(pretrained=True)
        # model =  InceptionResnetV1(num_classes=10178, pretrained='vggface2', device=run_helper.device, classify=True)
        # logger.error(model.num_classes)
        model.fc = nn.Linear(2048, 10178)
        run_helper.fixed_model = model

    elif run_helper.data  == 'nlp':
        from transformers import BertModel
        a = 0
        run_helper.load_text()
        print(f'Time to load: {0 - a }')
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = RNNModel(bert)
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        run_helper.fixed_model = model

    else:
        raise Exception('Specify dataset')


    run_helper.check_resume_training(model)
    model.to(run_helper.device)
    # if run_helper.smoothing:
    #     model = model[1]
    #     run_helper.fixed_model = run_helper.fixed_model[1]

    if run_helper.nc and not run_helper.new_nc_evasion:
        run_helper.mixed = Mixed(model, size=run_helper.train_dataset[0][0].shape[1])
        run_helper.mixed = run_helper.mixed.to(run_helper.device)
        run_helper.mixed_optim = torch.optim.Adam(helper.mixed.parameters(), lr=0.01)
    if run_helper.data == 'nlp':
        criterion = nn.BCEWithLogitsLoss().to(run_helper.device)
    elif run_helper.dp:
        criterion = nn.CrossEntropyLoss(reduction='none').to(run_helper.device)
    elif run_helper.data == 'celeba':
        run_helper.celeb_criterion = nn.TripletMarginLoss(margin=0.2, swap=True).to(run_helper.device)
        criterion = nn.CrossEntropyLoss().to(run_helper.device)
    else:
        criterion = nn.CrossEntropyLoss().to(run_helper.device)

    optimizer = run_helper.get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])
    scheduler._step_count = run_helper.start_epoch
    scheduler.last_epoch = run_helper.start_epoch

    # test(run_helper, model, criterion, epoch=0)
    # acc_p, loss_p = test(run_helper, model, criterion, epoch=0, is_poison=True)
    run_helper.total_times = list()

    for epoch in range(run_helper.start_epoch, run_helper.epochs+1):
        logger.error(epoch)
        logger.warning(optimizer.param_groups[0]["lr"])
        train(run_helper, model, optimizer, criterion, epoch=epoch)
        acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True)
        if not run_helper.timing:
            if run_helper.data=='multimnist':
                acc_p, loss_p = test(run_helper, model, criterion, epoch=epoch, is_poison=True, sum=True)
                run_helper.save_dict[f'acc.back'].append(acc_p)
            acc, loss = test(run_helper, model, criterion, epoch=epoch)
            run_helper.save_dict[f'acc'].append(acc)
            if run_helper.scheduler:
                scheduler.step()
            if run_helper.mixed:
                run_helper.save_mixed(epoch)
            run_helper.save_model(model, epoch, acc)

        if run_helper.timing:
            run_helper.total_times.append(np.mean(run_helper.times['total']))
            # if acc_p >=90:
            #     break
    if run_helper.timing == True:
        logger.error(run_helper.times)
    elif run_helper.timing == 'total':
        logger.error(run_helper.times['total'])
        logger.error([np.mean(run_helper.times['total'][1:]), np.std(run_helper.times['total'][1:]),  stats.sem(run_helper.times['total'][1:], axis=None, ddof=0)])
        logger.error(run_helper.total_times)

    if run_helper.memory:
        logger.warning(torch.cuda.memory_summary(abbreviated=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', required=True)

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    # if params['data'] == 'image':
    # helper = ImageHelper(current_time=d, params=params, name='image')
    # else:
    #     helper = TextHelper(current_time=d, params=params, name='text')
    #     helper.corpus = torch.load(helper.params['corpus'])
    #     logger.info(helper.corpus.train.shape)
    helper = Helper(params)
    train_new(helper)

    #
    # if helper.random_seed is not None:
    #     helper.fix_random(helper.random_seed)
    #
    # logger.error(yaml.dump(helper.params))
    # try:
    #     run(helper)
    #     if helper.is_save and len(helper.save_dict):
    #         torch.save(helper.save_dict,f'{helper.folder_path}/save_dict.pt')
    #     if helper.log:
    #         print(f'You can find files in {helper.folder_path}. TB graph: {args.name}')
    # except KeyboardInterrupt:
    #
    #     if helper.timing == True:
    #         logger.error(helper.times)
    #     elif helper.timing == 'total':
    #         logger.error(helper.times)
    #         logger.error([np.mean(helper.times['total'][1:]), np.std(helper.times['total'][1:])])
    #         logger.error(helper.total_times)
    #
    #     if helper.memory:
    #         logger.warning(torch.cuda.memory_summary(abbreviated=True))
    #     if helper.log:
    #         answer = prompt('\nDelete the repo? (y/n): ')
    #         if answer in ['Y', 'y', 'yes']:
    #             logger.error(f"Fine. Deleted: {helper.folder_path}")
    #             shutil.rmtree(helper.folder_path)
    #             if helper.tb:
    #                 shutil.rmtree(f'runs/{args.name}')
    #         else:
    #             torch.save(helper.save_dict, f'{helper.folder_path}/save_dict.pt')
    #             logger.error(f"Aborted training. Results: {helper.folder_path}. TB graph: {args.name}")
    #     else:
    #         logger.error(f"Aborted training. No output generated.")


