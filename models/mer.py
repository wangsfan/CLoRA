# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class Mer(ContinualModel):
    """Continual Learning via Meta-Experience Replay (Alg 6)."""
    NAME = 'mer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.set_defaults(batch_size=1)

        parser.add_argument('--beta', type=float, required=True,
                            help='Within-batch update beta parameter.')
        parser.add_argument('--gamma', type=float, required=True,
                            help='Across-batch update gamma parameter.')
        parser.add_argument('--batch_num', type=int, default=1,
                            help='Number of batches extracted from the buffer.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.batch_size != 1:
            logging.warning('MER is designed to work with batch_size=1. We will use batch_size=1.')
            args.batch_size = 1
        super(Mer, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        # 保存初始参数 θ_A0
        theta_A0 = torch.nn.utils.parameters_to_vector(
            self.net.parameters()
        ).detach().clone()

        for _ in range(self.args.batch_num):

            theta_Wi0 = torch.nn.utils.parameters_to_vector(
                self.net.parameters()
            ).detach().clone()

            # === 构造 batch ===
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size,
                    transform=self.transform,
                    device=self.device
                )
                batch_inputs = torch.cat((buf_inputs, inputs))
                batch_labels = torch.cat(
                    (buf_labels, torch.tensor([labels]).to(self.device))
                )
            else:
                batch_inputs = inputs
                batch_labels = torch.tensor([labels]).to(self.device)

            # === within-batch SGD ===
            self.opt.zero_grad()
            outputs = self.net(batch_inputs)
            loss = self.loss(outputs, batch_labels)
            loss.backward()
            self.opt.step()

            # === reptile within-batch meta update ===
            theta_Wi = torch.nn.utils.parameters_to_vector(
                self.net.parameters()
            )

            new_theta = theta_Wi0 + self.args.beta * (theta_Wi - theta_Wi0)

            torch.nn.utils.vector_to_parameters(new_theta, self.net.parameters())

        # 存入 buffer
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        # === across-batch reptile update ===
        theta_A = torch.nn.utils.parameters_to_vector(
            self.net.parameters()
        )

        new_theta = theta_A0 + self.args.gamma * (theta_A - theta_A0)

        torch.nn.utils.vector_to_parameters(new_theta, self.net.parameters())

        return loss.item()
