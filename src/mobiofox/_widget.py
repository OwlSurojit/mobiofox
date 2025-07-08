import napari
from magicgui import magic_factory
from magicgui.widgets import (
    PushButton,
)
from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QWidget

from ._pipeline_widgets import (
    CropWidget,
    FilterWidget,
    MorphometryWidget,
    SegmentWidget,
)
from ._utils._histogram import show_histogram
from ._utils._metadata_widget import LayerPickerWithMetadata


class MorphometryPipelineWidget(QWidget):

    STEP_DESCRIPTIONS = [
        "Cropping",
        "Filtering",
        "Segmentation",
        "Extract morphometric features",
    ]

    PIPELINE_WIDGET_TYPES = [
        CropWidget,
        FilterWidget,
        SegmentWidget,
        MorphometryWidget,
    ]

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._input_with_metadata = LayerPickerWithMetadata(viewer)

        self._input_image_picker = self._input_with_metadata.layer_picker

        viewer.layers.events.inserted.connect(
            self._input_image_picker.reset_choices
        )
        viewer.layers.events.removed.connect(
            self._input_image_picker.reset_choices
        )
        viewer.layers.events.reordered.connect(
            self._input_image_picker.reset_choices
        )

        self._next_step_button = PushButton(
            text=f"Start first step ({self.STEP_DESCRIPTIONS[0]})"
        )
        self._next_step_button.changed.connect(self._start_next_step)
        self._current_step = 0

        self._skip_button = PushButton(
            text=f"Skip step ({self.STEP_DESCRIPTIONS[0]})",
        )
        self._skip_button.changed.connect(self._skip_step)

        self._cur_widget = self._input_image_picker

        layout = QVBoxLayout()
        self.setLayout(layout)
        # layout.addWidget(self._input_image_picker.native)
        self._insert_groupbox(0, "Input data", self._input_with_metadata)
        layout.addWidget(self._next_step_button.native)
        layout.addWidget(self._skip_button.native)

    def _image_layer_choices(self, _widget):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, "napari.layers.Image")
        ]

    def _insert_groupbox(self, idx, title, container):
        box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.addWidget(container.native)
        box.setLayout(layout)
        self.layout().insertWidget(idx, box)

    def _update_button_texts(self):
        self._next_step_button.text = (
            f"Next step ({self.STEP_DESCRIPTIONS[self._current_step]})"
        )
        self._skip_button.text = (
            f"Skip step ({self.STEP_DESCRIPTIONS[self._current_step]})"
        )

    def _start_next_step(self):
        next_widget_type = self.PIPELINE_WIDGET_TYPES[self._current_step]
        if next_widget_type is MorphometryWidget:
            next_step_widget = next_widget_type(
                self._viewer,
                self._cur_widget,
                intensity_input_widget=self._input_image_picker,
            )
        else:
            next_step_widget = next_widget_type(self._viewer, self._cur_widget)
        self._cur_widget = next_step_widget

        idx = self.layout().indexOf(self._next_step_button.native)
        self._insert_groupbox(
            idx,
            f"Step {self._current_step+1}: {self.STEP_DESCRIPTIONS[self._current_step]}",
            next_step_widget,
        )
        self._current_step += 1

        if self._current_step < len(self.PIPELINE_WIDGET_TYPES):
            self._update_button_texts()
        else:
            # TODO not sure about this
            self.layout().removeWidget(self._next_step_button.native)
            self.layout().removeWidget(self._skip_button.native)

    def _skip_step(self):
        self._current_step += 1
        if (
            self._current_step < len(self.PIPELINE_WIDGET_TYPES) - 1
        ):  # -1 because MorphometryWidget relies on segmentation
            self._update_button_texts()
        else:
            self.layout().removeWidget(self._next_step_button.native)
            self.layout().removeWidget(self._skip_button.native)


@magic_factory(
    call_button="Show histogram",
)
def histogram_widget(img_layer: "napari.layers.Image") -> None:
    print("Histogram widget called")
    show_histogram(
        img_layer.data[img_layer.data != 0], layer_name=img_layer.name
    )
