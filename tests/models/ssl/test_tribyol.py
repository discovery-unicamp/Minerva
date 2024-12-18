import unittest
import torch
import torch.nn as nn
from minerva.models.ssl.tribyol import TriBYOL, MLPHead

class DummyBackbone(nn.Module):
    def __init__(self, output_dim=2048):
        super(DummyBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Conv2d(16, output_dim, kernel_size=1)  # Replace Linear with Conv2d

    def forward(self, x):
        x = self.conv(x)  # Output shape: (batch_size, 16, height, width)
        return self.fc(x)  # Output shape: (batch_size, output_dim, height, width)


class TestTriBYOL(unittest.TestCase):
    def setUp(self):
        """Set up the TriBYOL model with a dummy backbone for testing."""
        self.backbone = DummyBackbone()
        self.model = TriBYOL(backbone=self.backbone)
        self.batch_size = 4
        self.image_size = (3, 32, 32)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def generate_dummy_batch(self, batch_size, image_size):
        """Generate a batch of dummy images for testing."""
        return torch.randn(batch_size, *image_size)

    def test_forward_pass(self):
        """Test the forward pass of the TriBYOL model."""
        x1 = self.generate_dummy_batch(self.batch_size, self.image_size).to(self.device)
        x2 = self.generate_dummy_batch(self.batch_size, self.image_size).to(self.device)
        x3 = self.generate_dummy_batch(self.batch_size, self.image_size).to(self.device)

        # Perform forward pass
        p1, t2, t3 = self.model(x1, x2, x3)

        self.assertEqual(p1.shape, (self.batch_size, 128))
        self.assertEqual(t2.shape, (self.batch_size, 128))
        self.assertEqual(t3.shape, (self.batch_size, 128))

    def test_loss_function(self):
        """Test the loss function for proper output."""
        x = torch.randn(self.batch_size, 128)
        y = torch.randn(self.batch_size, 128)

        loss = self.model.loss_fn(x, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss

    def test_training_step(self):
        """Test the training step."""
        batch = [self.generate_dummy_batch(self.batch_size, self.image_size).to(self.device)]
        loss = self.model.training_step(batch, batch_idx=0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_update_target_networks(self):
        """Test if target networks are updated via EMA."""
        initial_params = [p.clone().detach() for p in self.model.target_network_1.parameters()]

        # Perform an optimizer step to trigger `update_target_networks`
        optimizer = self.model.configure_optimizers()
        batch = [self.generate_dummy_batch(self.batch_size, self.image_size).to(self.device)]
        loss = self.model.training_step(batch, batch_idx=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.model.update_target_networks()  # Explicitly call update for clarity

        updated_params = [p.clone().detach() for p in self.model.target_network_1.parameters()]

        # Debugging output
        for i, (initial, updated) in enumerate(zip(initial_params, updated_params)):
            print(f"Param {i} - Initial: {initial.mean().item()}, Updated: {updated.mean().item()}")

        for initial, updated in zip(initial_params, updated_params):
            self.assertFalse(torch.equal(initial, updated))


    def test_ensure_tensor(self):
        """Test the ensure_tensor method for proper conversion to tensor."""
        from PIL import Image
        import numpy as np

        image = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        tensor = self.model.ensure_tensor(image)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
