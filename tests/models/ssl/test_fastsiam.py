import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from minerva.models.ssl.fastsiam import FastSiam, SimSiamMLPHead

class TestFastSiam(unittest.TestCase):
    def setUp(self):
        # Mock backbone
        self.mock_backbone = MagicMock(spec=nn.Module)
        self.mock_backbone.return_value = torch.rand(4, 2048, 1, 1)  # Mock output of backbone

        # Initialize FastSiam
        self.model = FastSiam(
            backbone=self.mock_backbone,
            in_dim=2048,
            hid_dim=512,
            out_dim=128,
            K=2,
            momentum=0.996,
            lr=1e-3,
        )

    def test_forward(self):
        # Test the forward pass
        views = [torch.rand(4, 3, 224, 224) for _ in range(3)]  # Mock 3 augmented views
        prediction, target = self.model(views)
        
        # Assertions
        self.assertEqual(prediction.shape, (4, 128))
        self.assertEqual(target.shape, (4, 128))

    def test_update_target_branch(self):
        # Test momentum update of target branch
        original_params = [p.clone() for p in self.model.target_branch_backbone.parameters()]
        self.model.update_target_branch()
        updated_params = [p for p in self.model.target_branch_backbone.parameters()]

        for orig, updated in zip(original_params, updated_params):
            self.assertFalse(torch.equal(orig, updated), "Target branch parameters did not update correctly.")

    def test_fastsiam_loss(self):
        # Test the FastSiam loss function
        pred = torch.rand(4, 128)
        target = torch.rand(4, 128)
        loss = self.model.fastsiam_loss(pred, target)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_training_step(self):
        # Mock batch
        batch = (torch.rand(4, 3, 224, 224),)

        # Run training step
        loss = self.model.training_step(batch, 0)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_configure_optimizers(self):
        # Mock the Trainer
        mock_trainer = MagicMock()
        mock_trainer.max_epochs = 10  # or any integer value you want to simulate
        self.model.trainer = mock_trainer

        # Test optimizer and scheduler configuration
        optimizers, schedulers = self.model.configure_optimizers()
        self.assertEqual(len(optimizers), 1)
        self.assertEqual(len(schedulers), 1)
        self.assertIsInstance(optimizers[0], torch.optim.SGD)
        self.assertIsInstance(schedulers[0], torch.optim.lr_scheduler.CosineAnnealingLR)


    def test_mlp_head(self):
        # Test the SimSiamMLPHead
        mlp = SimSiamMLPHead([128, 256, 128], activation_cls=nn.ReLU, batch_norm=True, final_bn=True, final_relu=False)
        input_tensor = torch.rand(4, 128)
        output_tensor = mlp(input_tensor)

        # Assertions
        self.assertEqual(output_tensor.shape, (4, 128))

if __name__ == "__main__":
    unittest.main()
