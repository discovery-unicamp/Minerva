import unittest
import torch
import torch.nn as nn
from minerva.models.ssl.fastsiam import FastSiam, MLPHead


# Mock backbone for testing purposes
class MockBackbone(nn.Module):
    def __init__(self, output_dim=2048):
        super(MockBackbone, self).__init__()
        self.output_dim = output_dim
        self.conv = nn.Conv2d(3, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Simple convolution followed by global average pooling
        return self.conv(x)


class TestFastSiam(unittest.TestCase):

    def setUp(self):
        """Set up a FastSiam model with a mock backbone for each test."""
        self.backbone = MockBackbone(output_dim=2048)
        self.model = FastSiam(backbone=self.backbone)

    def test_mlp_head(self):
        """Test the MLPHead for correct output dimensions."""
        mlp = MLPHead(input_dim=512, hidden_dim=256, output_dim=128)
        x = torch.randn(4, 512)  # Batch of 4, feature size 512
        output = mlp(x)
        self.assertEqual(output.shape, (4, 128), f"Expected output shape (4, 128), got {output.shape}")

    def test_fastsiam_initialization(self):
        """Test if the FastSiam model initializes properly."""
        self.assertIsInstance(self.model.backbone, nn.Module, "Backbone is not an instance of nn.Module")
        self.assertIsInstance(self.model.student_projector, MLPHead, "Student projector is not an MLPHead")
        self.assertIsInstance(self.model.student_predictor, MLPHead, "Student predictor is not an MLPHead")
        self.assertIsInstance(self.model.teacher_backbone, nn.Module, "Teacher backbone is not an instance of nn.Module")
        self.assertIsInstance(self.model.teacher_projector, MLPHead, "Teacher projector is not an MLPHead")

    def test_forward_pass(self):
        """Test the forward pass of the FastSiam model."""
        # Create a batch of 2 images (3 channels, 224x224)
        views = [torch.randn(2, 3, 224, 224) for _ in range(4)]  # 4 views for K=3 (3 teacher + 1 student)

        student_predicted, avg_teacher_projected = self.model(views)

        # Check the output dimensions
        self.assertEqual(student_predicted.shape, (2, 128), f"Expected student_predicted shape (2, 128), got {student_predicted.shape}")
        self.assertEqual(avg_teacher_projected.shape, (2, 128), f"Expected avg_teacher_projected shape (2, 128), got {avg_teacher_projected.shape}")

    def test_teacher_update(self):
        """Test the teacher network's parameters are updated correctly with momentum."""
        # Get a snapshot of the teacher parameters before update
        initial_teacher_params = [p.clone().detach() for p in self.model.teacher_backbone.parameters()]

        # Perform a forward pass to simulate training
        views = [torch.randn(2, 3, 224, 224) for _ in range(4)]
        self.model(views)

        # Update the teacher network
        self.model.update_teacher()

        # Ensure teacher parameters have changed
        for initial, updated in zip(initial_teacher_params, self.model.teacher_backbone.parameters()):
            self.assertFalse(torch.equal(initial, updated), "Teacher parameters were not updated")

    def test_single_step_loss(self):
        """Test if the _single_step method returns a valid loss value."""
        images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        batch = (images,)

        loss = self.model._single_step(batch, self.model.K, log_prefix="train")
        self.assertIsInstance(loss, torch.Tensor, "Loss should be a torch.Tensor")
        self.assertGreaterEqual(loss.item(), 0, "Loss should be non-negative")

    def test_configure_optimizers(self):
        """Test if the optimizer is configured correctly."""
        optimizer = self.model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.Adam, "Optimizer should be an instance of torch.optim.Adam")
        self.assertEqual(optimizer.defaults["lr"], self.model.lr, f"Expected learning rate {self.model.lr}, got {optimizer.defaults['lr']}")


if __name__ == "__main__":
    unittest.main()
