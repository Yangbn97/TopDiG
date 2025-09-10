# from tensorboardX import SummaryWriter
import errno
import os
import sys
import config_file as cfg_file
import datetime
from torch.utils.tensorboard import SummaryWriter


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


class LogSummary(object):

    def __init__(self, log_path):

        mkdirs(log_path)
        self.writer = SummaryWriter(log_path)

    def write_scalars(self, scalar_dict, n_iter, tag=None):

        for name, scalar in scalar_dict.items():
            if tag is not None:
                name = '/'.join([tag, name])
            self.writer.add_scalar(name, scalar, n_iter)

    def write_hist_parameters(self, net, n_iter):
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().detach().numpy(), n_iter)
            self.writer.add_histogram(name + '/grad', param.grad.clone().data.cpu().numpy(), n_iter)


def make_print_to_file(env):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    path = env.record_root

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    if env.configs['Experiment']['evaluate']:
        fileName = datetime.datetime.now().strftime('L' + str(env.configs['Model']['num_attention_layers']) + 'H' + str(env.configs['Model']['num_heads']) + '_eval' + ' | ' + 'day' + '%Y-%m-%d %H:%M:%S')
    elif env.configs['Experiment']['infer']:
        fileName = datetime.datetime.now().strftime('L' + str(env.configs['Model']['num_attention_layers']) + 'H' + str(env.configs['Model']['num_heads']) + '_infer' + ' | ' + 'day' + '%Y-%m-%d %H:%M:%S')
    elif env.configs['Experiment']['pretrain']:
        fileName = datetime.datetime.now().strftime('pretrain' + ' | ' + 'day' + '%Y-%m-%d %H:%M:%S')
    else:
        fileName = datetime.datetime.now().strftime('L' + str(env.configs['Model']['num_attention_layers']) + 'H' + str(env.configs['Model']['num_heads']) + 'train' + ' | ' + '%Y-%m-%d %H:%M:%S')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))
