from torch import nn, optim
import lightning as L
import torch

class CPC(L.LightningModule):
    """Implements the Contrastive Predictive Coding (CPC) model, as described in
    https://dl.acm.org/doi/10.1145/3463506. The implementation was adapted from
    https://github.com/harkash/contrastive-predictive-coding-for-har
    """
    def __init__(
        self,
        g_enc: L.LightningModule,
        g_ar: L.LightningModule,
        prediction_head: L.LightningModule,
        num_steps_prediction: int = 28,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ):  
        super(CPC, self).__init__()
        self.g_enc = g_enc
        self.g_ar = g_ar
        self.prediction_head = prediction_head
        self.predictors = nn.ModuleList([self.prediction_head
                                 for i in range(num_steps_prediction)])
        self.learning_rate = learning_rate
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.batch_size = batch_size
        self.num_steps_prediction = num_steps_prediction

    def forward(self, x):
        z = self.g_enc(x)
        r_out, _ = self.g_ar(z, None)
        r_out = r_out[:, -1, :]
        y = torch.stack([pred(r_out) for pred in self.predictors], dim=1)
        # print("Y", y.shape)
        # print("Z", z.shape)
        return z, y
    
    def training_step(self, batch, batch_idx):
        z, _ = self.forward(batch[0])

        batch_size = batch[0].size(0) # 64

        S = z.size(1)

        # Tempo aleatório para começar a previsão futura

        start = torch.randint(high=int(S - self.num_steps_prediction),
                              size=(1,), low=2).long()
                
        # Contexto : Dados de entrada até o tempo inicial (PASSADO)

        context = batch[0][:, :, :start+1]

        # Verdade : Dados de entrada até o tempo inicial + o número de passos de previsão (FUTURO)

        y_truth = z[:, start+1:start+1+self.num_steps_prediction, :].permute(1, 0, 2)

        _, y_pred = self.forward(context)

        y_pred = y_pred.permute(1, 0, 2)

        nce = 0
        correct = 0
        correct_steps = []

        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):
            log_density_ratio = torch.mm(y_truth[k],y_pred[k].transpose(0, 1))

            # correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio),
                                               dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)

            correct = (correct +
                       torch.sum(torch.eq(positive_batch_pred,
                                          positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred,
                                                    positive_batch_actual)).item())

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        self.log("train_loss", nce)
        return nce

    def validation_step(self, batch, batch_idx):
        z, _ = self.forward(batch[0]) 

        batch_size = batch[0].size(0)

        S = z.size(1)

        start = torch.randint(high=int(S - self.num_steps_prediction),low=2,
                              size=(1,)).long()
        
        # Contexto : Dados de entrada até o tempo inicial (PASSADO)

        context = batch[0][:, :, :start+1] #64, 6, tudo até start+1

        # Verdade : Dados de entrada até o tempo inicial + o número de passos de previsão (FUTURO)

        y_truth = z[:, start+1:start+1+self.num_steps_prediction, :].permute(1, 0, 2) 

        _, y_pred = self.forward(context)

        y_pred = y_pred.permute(1, 0, 2)

        nce = 0
        correct = 0
        correct_steps = []

        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):

            log_density_ratio = torch.mm(y_truth[k],y_pred[k].transpose(0, 1))

            # correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio),
                                               dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)

            correct = (correct +
                       torch.sum(torch.eq(positive_batch_pred,
                                          positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred,
                                                    positive_batch_actual)).item())

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        self.log("val_loss", nce)
        return nce

    def test_step(self, batch, batch_idx):
        z, _ = self.forward(batch[0]) 

        batch_size = batch[0].size(0) # 64

        S = z.size(1)

        start = torch.randint(high=int(S - self.num_steps_prediction),low=2,
                              size=(1,)).long()
        
        # Contexto : Dados de entrada até o tempo inicial (PASSADO)

        context = batch[0][:, :, :start+1] #64, 6, tudo até start+1

        # Verdade : Dados de entrada até o tempo inicial + o número de passos de previsão (FUTURO)

        y_truth = z[:, start+1:start+1+self.num_steps_prediction, :].permute(1, 0, 2) 

        _, y_pred = self.forward(context)

        y_pred = y_pred.permute(1, 0, 2)

        nce = 0
        correct = 0
        correct_steps = []

        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):

            log_density_ratio = torch.mm(y_truth[k],y_pred[k].transpose(0, 1))

            # correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio),
                                               dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)

            correct = (correct +
                       torch.sum(torch.eq(positive_batch_pred,
                                          positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred,
                                                    positive_batch_actual)).item())

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        self.log("test_loss", nce)
        return nce

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
