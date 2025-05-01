from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui.widgets import Widget

from . import PipelineWidget


class FilterWidget(PipelineWidget):
    """
    Widget to apply various filters and remove artifacts from the scanning process.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        input_widget: Widget | None = None,
    ):
        super().__init__(viewer, input_widget, "_filtered")
