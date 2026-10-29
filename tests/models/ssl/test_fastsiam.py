import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from minerva.models.ssl.fastsiam import FastSiam, SimSiamMLPHead


class TestFastSiam(unittest.TestCase):
    def setUp(self):
        # Mock backbone
        self.mock_backbone = MagicMock(spec=nn.Module)
        self.mock_backbone.return_value = torch.rand(
            4, 2048, 1, 1
        )  # Mock output of backbone

        # Initialize FastSiam
        self.model = FastSiam(
            backbone=self.mock_backbone,
            in_dim=2048,
            hid_dim=512,
            out_dim=128,
            k=2,  # Using k=2 means the batch should contain k+1=3 views
            lr=0.125,
        )

    def test_forward(self):
        # Test the forward pass
        view = torch.rand(
            4, 3, 224, 224
        )  # Mock 1 augmented view (forward takes one view)
        z, p = self.model(view)

        # Assertions
        self.assertEqual(z.shape, (4, 128))
        self.assertEqual(p.shape, (4, 128))
        self.assertFalse(z.requires_grad)
        self.assertTrue(p.requires_grad)

    def test_single_step_arbitrary_k(self):
        # Test the arbitrary k step function
        # Mock 3 augmented views for k=2
        batch = tuple([torch.rand(4, 3, 224, 224) for _ in range(3)])

        # Run single step
        loss = self.model._single_step_arbitrary_k(batch)

        # Assertions
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_training_step(self):
        # Mock batch (k=2 requires 3 views)
        batch = tuple([torch.rand(4, 3, 224, 224) for _ in range(3)])

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
        mlp = SimSiamMLPHead(
            [128, 256, 128],
            activation_cls=nn.ReLU,
            batch_norm=True,
            final_bn=True,
            final_relu=False,
        )
        input_tensor = torch.rand(4, 128)
        output_tensor = mlp(input_tensor)

        # Assertions
        self.assertEqual(output_tensor.shape, (4, 128))

    def test_unexpected_k_error(self):
        batch = tuple([torch.rand(4, 3, 224, 224) for _ in range(2)])
        with self.assertRaisesRegex(
            RuntimeError,
            r"expected 3 views, but got 2, is your Dataset class yielding k\+1 views\?",
        ):
            self.model.training_step(batch, 0)

    def test_single_step_k_equals_3(self):
        model_k3 = FastSiam(backbone=self.mock_backbone, k=3)
        batch = tuple([torch.rand(4, 3, 224, 224) for _ in range(4)])
        loss = model_k3.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)

    def test_avg_pooling_flag(self):
        model = FastSiam(
            backbone=self.mock_backbone,
            in_dim=2048,
            hid_dim=512,
            out_dim=128,
            avg_pooling=False,
        )
        self.assertIsNone(model.global_avg_pool)

        view = torch.rand(4, 3, 224, 224)
        z, p = model(view)

        self.assertEqual(z.shape, (4, 128))
        self.assertEqual(p.shape, (4, 128))

    def test_flatten_flag(self):
        mock_backbone = MagicMock(spec=nn.Module)
        mock_backbone.return_value = torch.rand(4, 2048)

        model = FastSiam(
            backbone=mock_backbone,
            in_dim=2048,
            hid_dim=512,
            out_dim=128,
            flatten=False,
            avg_pooling=False,
        )

        self.assertFalse(model.flatten)

        view = torch.rand(4, 3, 224, 224)
        z, p = model(view)

        self.assertEqual(z.shape, (4, 128))
        self.assertEqual(p.shape, (4, 128))


if __name__ == "__main__":
    unittest.main()
