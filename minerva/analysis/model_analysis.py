from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import lightning as L
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
from minerva.models.nets.base import SimpleSupervisedModel

import torch

from minerva.data.data_module_tools import get_full_data_split
from minerva.utils.typing import PathLike

import plotly.graph_objects as go

# Global variable to control plotly.js inclusion
_plot_tsne_written_dirs = set()


class _ModelAnalysis:
    """Main interface for model analysis. A model analysis is a post-training
    analysis that can be run on a trained model to generate insights about the
    model's performance. It has a `path` attribute that specifies the directory
    where the analysis results will be saved. The `compute` method should be
    implemented by subclasses to perform the actual analysis. All insights
    generated by the analysis should be saved in the `path` directory.
    Note that, differently from `Metric`, `_ModelAnalysis` does not return any
    value. Instead, the results of the analysis should be saved in the `path`
    directory. All subclasses of `_ModelAnalysis` should implement the `compute`
    method. Inside a pipeline the path will be automatically set to the
    `pipeline.log_dir` attribute.
    """

    def __init__(self, path: Optional[PathLike] = None):
        self._path = path

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: PathLike):
        self._path = Path(path)

    def compute(self, model: L.LightningModule, data: L.LightningDataModule):
        raise NotImplementedError


class TSNEAnalysis:
    """Perform t-SNE analysis on the embeddings generated by a model.
    A t-SNE plot is generated using the embeddings and saved in the `path`
    directory. The plot is saved as both PNG and HTML files.
    """

    def __init__(
        self,
        label_names: Optional[Dict[Union[int, str], str]] = None,
        height: int = 800,
        width: int = 800,
        text_size: int = 12,
        title: Optional[str] = None,
        x_axis_title: str = "x",
        y_axis_title: str = "y",
        legend_title: str = "Label",
        output_filename: PathLike = "tsne.png",
        seed: int = 42,
        n_components: int = 2,
        marker_symbols: Optional[Dict[str, str]] = None,
        colors: Optional[Dict[str, str]] = None,
        show_legend: bool = True,
        show_tick_labels: bool = True,
        title_font_scale: float = 1.5,
        tick_font_scale: float = 1.2,
        title_y_position: float = 0.95,
        write_html: bool = True,
    ):
        """Plot a t-SNE plot of the embeddings generated by a model.

        Parameters
        ----------
        label_names : Optional[Dict[Union[int, str], str]], optional
            Labels to use for the plot, instead of the original labels in the
            data (`y`). The keys are the original labels and the values are the
            new labels to use in the plot. If None, the original labels are used
            as they are. By default None
        height : int, optional
            Height of the figure, by default 800
        width : int, optional
            Width of the figure, by default 800
        text_size : int, optional
            Size of font used in plot, by default 12
        title : str, optional
            Title of graph, by default None
        x_axis_title : str, optional
            Name of x-axis, by default "x"
        y_axis_title : str, optional
            Name of y-axis, by default "y"
        legend_title : str, optional
            Name for legend title, by default "Label"
        output_filename : PathLike, optional
            Name of the output file to save the plot as a PNG image file. The
            file will be saved in the `path` directory with this name. By
            default "tsne.png"
        seed : int, optional
            Random seed for t-SNE, by default 42
        n_components : int, optional
            Number of components to use in t-SNE, by default 2
        marker_symbols : Optional[Dict[str, str]], optional
            Dictionary mapping labels to marker symbols. If None, will use default symbols.
        colors : Optional[Dict[str, str]], optional
            Dictionary mapping labels to color values (hex codes or names). If None, will use plotly's default color sequence.
        show_legend : bool, optional
            If True, displays the legend. Default is True.
        show_tick_labels : bool, optional
            If True, axis tick labels are shown. Default is True.
        title_font_scale : float, optional
            Scaling factor applied to the title font size. Default is 1.5.
        tick_font_scale : float, optional
            Scaling factor applied to axis tick font sizes. Default is 1.2.
        title_y_position : float, optional
            Relative y-position of the title, used to adjust spacing above the plot. Default is 0.95.
        write_html: bool, optional
            If True, saves the plot as an HTML file with interactive controls. Default is True.
        """
        self.label_names = label_names
        self.height = height
        self.width = width
        self.text_size = text_size
        self.title = title
        self.output_filename = Path(output_filename)
        self.x_axis_title = x_axis_title
        self.y_axis_title = y_axis_title
        self.legend_title = legend_title
        self.seed = seed
        self.n_components = n_components
        self.path = None
        self.show_legend = show_legend
        self.show_tick_labels = show_tick_labels
        self.title_font_scale = title_font_scale
        self.tick_font_scale = tick_font_scale
        self.title_y_position = title_y_position

        # Store the provided marker symbols and colors (will be processed during plotting)
        self.marker_symbols = marker_symbols
        self.colors = colors
        self.write_html = write_html

        assert (
            self.n_components == 2
        ), "For now, n_components must be set to 2 for t-SNE analysis"

    def set_path(self, path: PathLike):
        """Set the output path for saving the plots."""
        self.path = Path(path)

    def _get_default_marker_symbols(self, labels: List[str]) -> Dict[str, str]:
        """Generate default marker symbols for each label."""
        # Basic set of marker symbols that will cycle through
        default_symbols = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "star",
            "hexagram",
            "pentagon",
        ]
        return {
            label: default_symbols[i % len(default_symbols)]
            for i, label in enumerate(sorted(labels))
        }

    def _get_default_colors(self, labels: List[str]) -> Dict[str, str]:
        """Generate default colors for each label using plotly's color sequence."""
        color_sequence = px.colors.qualitative.Plotly
        return {
            label: color_sequence[i % len(color_sequence)]
            for i, label in enumerate(sorted(labels))
        }

    def compute(
        self, model: SimpleSupervisedModel, data: L.LightningDataModule
    ) -> Dict[str, Any]:
        """Compute the t-SNE analysis and save the plot.

        Parameters
        ----------
        model : SimpleSupervisedModel
            The trained model from which to extract embeddings.
        data : L.LightningDataModule
            The data module containing the dataset to analyze.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing paths to the saved PNG and HTML files, and the DataFrame with t-SNE results.
            The keys are:
            - "png_path": Path to the saved PNG file.
            - "html_path": Path to the saved HTML file (if `write_html` is True).
            - "tnse_df": DataFrame containing the t-SNE results with columns "x", "y", and "label".
        """
        if not self.path:
            raise ValueError(
                "Path is not set. Please set the path before running the analysis."
            )

        model.eval()
        X, y = get_full_data_split(data, "predict")
        X = torch.tensor(X, device="cpu")
        embeddings = model.backbone.forward(X)  # type: ignore
        embeddings = embeddings.flatten(start_dim=1).detach().cpu().numpy()

        # Perform t-SNE on embeddings
        tsne_embeddings = TSNE(
            n_components=self.n_components, random_state=self.seed
        ).fit_transform(embeddings)

        # Create a DataFrame with embeddings and labels
        df = pd.DataFrame(data=tsne_embeddings, columns=["x", "y"])
        df["label"] = y

        # If label names are provided, map the original labels to the new labels
        if self.label_names is not None:
            df["label"] = df["label"].map(self.label_names)

        # Convert all labels to strings and sort
        df["label"] = df["label"].astype(str)
        df = df.sort_values(by="label")

        # Get unique labels
        unique_labels = sorted(df["label"].unique())

        # Generate default marker symbols and colors if not provided
        marker_symbols = (
            self.marker_symbols
            if self.marker_symbols is not None
            else self._get_default_marker_symbols(unique_labels)
        )
        colors = (
            self.colors
            if self.colors is not None
            else self._get_default_colors(unique_labels)
        )

        # Create figure using graph_objects
        fig = go.Figure()

        # Add traces for each label
        for label in unique_labels:
            df_label = df[df["label"] == label]
            fig.add_trace(
                go.Scatter(
                    x=df_label["x"],
                    y=df_label["y"],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=8,
                        symbol=marker_symbols.get(label, "circle"),
                        color=colors.get(label),
                        line=dict(width=2),
                    ),
                )
            )

        # Customize layout
        fig.update_layout(
            height=self.height,
            width=self.width,
            legend_title_text=self.legend_title,
            xaxis_title=None if not self.show_tick_labels else self.x_axis_title,
            yaxis_title=None if not self.show_tick_labels else self.y_axis_title,
            title={
                "text": self.title,
                "y": self.title_y_position,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=self.text_size * self.title_font_scale),
            },
            font=dict(size=self.text_size, family="Times New Roman"),
            margin=dict(l=10, r=10, t=80, b=10),
            legend=dict(
                orientation="h", y=1.1, x=0.5, xanchor="center", yanchor="bottom"
            ),
            showlegend=self.show_legend,
            xaxis=dict(
                showticklabels=self.show_tick_labels,
                tickfont=dict(size=self.text_size * self.tick_font_scale),
            ),
            yaxis=dict(
                showticklabels=self.show_tick_labels,
                tickfont=dict(size=self.text_size * self.tick_font_scale),
            ),
        )

        # Save the figure as PNG
        png_path = (self.path / self.output_filename).resolve()
        fig.write_image(png_path)
        print(f"✔ t-SNE plot saved to: {png_path}")

        # Additionally save as HTML with plotly.js control
        if self.write_html:
            html_path = png_path.with_suffix(".html")
            html_path.parent.mkdir(parents=True, exist_ok=True)
            dir_key = str(html_path.parent)
            include_js = dir_key not in _plot_tsne_written_dirs
            if include_js:
                _plot_tsne_written_dirs.add(dir_key)

            fig.write_html(str(html_path), include_plotlyjs=include_js, full_html=True)
            print(f"✔ HTML saved to: {html_path}")

        return {
            "png_path": str(png_path),
            "html_path": str(html_path) if self.write_html else None,
            "tnse_df": df,
        }
