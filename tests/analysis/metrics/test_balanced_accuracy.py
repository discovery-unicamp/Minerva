import torch
from minerva.analysis.metrics import BalancedAccuracy

def test_balanced_accuracy():
    # Test case for binary classification
    y_true_binary = torch.tensor([0, 1, 0, 0, 1, 0])
    y_pred_binary = torch.tensor([0, 1, 0, 0, 0, 1])
    num_classes_binary = 2
    task_binary = 'binary'
    expected_output_binary = 0.625

    metric_binary = BalancedAccuracy(num_classes=num_classes_binary, task=task_binary)
    metric_binary.update(y_pred_binary, y_true_binary)
    output_binary = metric_binary.compute()
    
    print(expected_output_binary, output_binary)

    assert (
        abs(output_binary - expected_output_binary) < 1e-6
    ), f"Expected output {expected_output_binary}, but got {output_binary}"

    # Test case for multiclass classification
    y_true_multiclass = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred_multiclass = torch.tensor([0, 2, 1, 0, 0, 1])
    num_classes_multiclass = 3
    task_multiclass = 'multiclass'
    expected_output_multiclass = 1 / 3  # In this example, only one class is predicted correctly

    metric_multiclass = BalancedAccuracy(num_classes=num_classes_multiclass, task=task_multiclass)
    metric_multiclass.update(y_pred_multiclass, y_true_multiclass)
    output_multiclass = metric_multiclass.compute()
    
    print(expected_output_multiclass, output_multiclass)

    assert (
        abs(output_multiclass - expected_output_multiclass) < 1e-6
    ), f"Expected output {expected_output_multiclass}, but got {output_multiclass}"