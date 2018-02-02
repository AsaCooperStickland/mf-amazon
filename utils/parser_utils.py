class ParserClass(object):
    def __init__(self, parser):
        """
        Parses arguments and saves them in the Parser Class
        :param parser: A parser to get input from
        """
        parser.add_argument('--batch_size', nargs="?", type=int, default=256, help='batch_size for experiment')
        parser.add_argument('--epochs', type=int, nargs="?", default=100, help='Number of epochs to train for')
        parser.add_argument('--logs_path', type=str, nargs="?", default="logs/",
                            help='Experiment log path, '
                                 'where tensorboard is saved, '
                                 'along with .csv of results')
        parser.add_argument('--experiment_prefix', nargs="?", type=str, default="mf",
                            help='Experiment name without hp details')
        parser.add_argument('--continue_epoch', nargs="?", type=int, default=-1, help="ID of epoch to continue from, "
                                                                                      "-1 means from scratch")
        parser.add_argument('--tensorboard_use', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--seed', nargs="?", type=int, default=1122017, help='Whether to use tensorboard')
        parser.add_argument('--l2_weight', nargs="?", type=float, default=1.0, help='l2 reg weight')
        parser.add_argument('--latent_dim', nargs="?", type=int, default=25, help='Whether to use tensorboard')
        parser.add_argument('--learning_rate', nargs="?", type=float, default=0.1, help='learning rate')

        self.args = parser.parse_args()

    def get_argument_variables(self):
        """
        Processes the parsed arguments and produces variables of specific types needed for the experiments
        :return: Arguments needed for experiments
        """
        batch_size = self.args.batch_size
        experiment_prefix = self.args.experiment_prefix
        seed = self.args.seed
        tensorboard_enable = True if self.args.tensorboard_use == "True" else False
        continue_from_epoch = self.args.continue_epoch  # use -1 to start from scratch
        epochs = self.args.epochs
        logs_path = self.args.logs_path
        l2_weight = self.args.l2_weight
        latent_dim = self.args.latent_dim
        learning_rate = self.args.learning_rate

        return batch_size, seed, epochs, logs_path, continue_from_epoch,\
            tensorboard_enable, experiment_prefix, l2_weight, latent_dim,\
            learning_rate
