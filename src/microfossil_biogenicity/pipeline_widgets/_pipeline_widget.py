from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui.widgets import Container, Widget, create_widget
from psygnal import Signal


class PipelineWidget(Container):
    """
    Base class for all widgets that can act as part of the microfossil biogenicity critera extraction pipeline.
    """

    changed = Signal(
        object, description="Emitted when the pipeline output changes."
    )

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        input_widget: Widget | None = None,
        output_layer_suffix: str = "_",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._viewer = viewer
        if input_widget is not None:
            self._input_widget = input_widget
        else:
            self._input_widget = create_widget(
                label="Input Data", annotation="napari.layers.Image"
            )
            self.append(self._input_widget)

        self.input_data = None
        self._output_layer_suffix = output_layer_suffix
        self._input_widget.changed.connect(self._input_changed)

    def _add_output_layer(self):
        image_layer = self._input_widget.value
        if image_layer is None:
            return False

        output_layer_name = image_layer.name + self._output_layer_suffix
        if output_layer_name not in self._viewer.layers:
            self._viewer.add_image(self.input_data, name=output_layer_name)
            image_layer.visible = False
        self._output_layer = self._viewer.layers[output_layer_name]

    def _input_changed(self):
        image_layer = self._input_widget.value
        if image_layer is None:
            return False
        self.input_data = image_layer.data

        if self.visible:
            self._add_output_layer()

        return True

    @property
    def output_data(self):
        return self.value.data

    @output_data.setter
    def output_data(self, value):
        if self.value is None:
            return
        self.value.data = value
        self.changed.emit(self.value)

    # Alias for the output layer
    @property
    def value(self):
        return getattr(self, "_output_layer", None)

    @value.setter
    def value(self, value):
        self._output_layer = value

    # Override visible property to add output layer to the viewer
    @property
    def visible(self):
        return super().visible

    @visible.setter
    def visible(self, value):
        super(PipelineWidget, self.__class__).visible.fset(
            self, value
        )  # Call the parent setter
        if value:
            self._add_output_layer()
