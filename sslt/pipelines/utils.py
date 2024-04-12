from io import BytesIO
from typing import Dict
import fsspec
import mlflow

import lightning as L
import torch


def tags2str(d: Dict[str, str]) -> str:
    """
    Convert a dictionary of tags to a search string compatible with MLflow's search_model_versions method.

    Parameters:
    - d: A dictionary containing tags where keys are tag names and values are tag values.

    Returns:
    - search_str: A search string formatted for MLflow's search_model_versions method.
    """
    search_str = " and ".join(
        [f"tags.`{key}`='{value}'" for key, value in d.items()]
    )
    return search_str


def load_model_mlflow(
    client: mlflow.client.MlflowClient,
    registered_model_name: str,
    registered_model_tags: Dict[str, str] = None,
) -> Dict[L.LightningModule, Dict[str, str]]:
    search_string = f"name='{registered_model_name}'"
    if registered_model_tags is not None:
        search_string += " and " + tags2str(registered_model_tags)

    registered_model = client.search_model_versions(
        search_string, order_by=["creation_timestamp DESC"], max_results=1
    )

    if len(registered_model) == 0:
        raise ValueError(
            f"No model found with the name '{registered_model_name}' and tags '{registered_model_tags}'. Query string used: {search_string}"
        )

    model_version = registered_model[0]
    run_id = model_version.run_id
    artifact_path = "/".join(model_version.source.split("/")[2:])
    artifact_uri = (
        client.get_run(run_id).info.artifact_uri + "/" + artifact_path
    )

    # print(f"Loading model from: {artifact_uri}")

    with fsspec.open(artifact_uri, "rb") as f:
        model_bytes = f.read()
        model_bytes = BytesIO(model_bytes)
        model = torch.load(model_bytes)

    print(f"Model loaded from: {artifact_uri}.")
    return model, dict(model_version)
