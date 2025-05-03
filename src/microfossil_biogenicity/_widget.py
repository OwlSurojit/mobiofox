from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
    create_widget,
)
from skimage.util import img_as_float

from .pipeline_widgets import CropWidget, FilterWidget


class MorphometryPipelineWidget(Container):

    STEP_DESCRIPTIONS = [
        "Crop",
        "Filter & remove artifacts",
        "Segment inclusions",
        "Extract morphometry features",
    ]

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._input_image_picker = create_widget(
            label="Input", annotation="napari.layers.Image"
        )

        # Caching intermediate steps
        self._input_image = None
        self._cropped = None
        self._filtered = None
        self._segmented = None

        self._next_step_button = PushButton(
            text=f"Start first step ({self.STEP_DESCRIPTIONS[0]})"
        )
        self._next_step_button.changed.connect(self._start_next_step)
        self._current_step = 0

        # Step 1: Crop
        self.crop_widget = CropWidget(
            viewer, self._input_image_picker, visible=False
        )
        # Step 2: Filter & remove artifacts
        self.filter_widget = FilterWidget(
            viewer, self.crop_widget, visible=False
        )
        # Step 3: Segment inclusions
        # Step 4: Extract morphometry features

        self.extend(
            [
                self._input_image_picker,
                self._next_step_button,
                Label(label="Step 1", value=self.STEP_DESCRIPTIONS[0]),
                self.crop_widget,
                Label(label="Step 2", value=self.STEP_DESCRIPTIONS[1]),
                self.filter_widget,
                Label(label="Step 3", value=self.STEP_DESCRIPTIONS[2]),
                Label(label="Step 4", value=self.STEP_DESCRIPTIONS[3]),
            ]
        )

    def _start_next_step(self):
        self._current_step += 1
        if self._current_step == 1:
            next_step_widget = self.crop_widget
        elif self._current_step == 2:
            next_step_widget = self.filter_widget
        else:
            return

        # Show new widgets before the button
        self.remove(self._next_step_button)
        next_step_widget.show()
        if self._current_step < len(self.STEP_DESCRIPTIONS):
            self._next_step_button.text = (
                f"Next step ({self.STEP_DESCRIPTIONS[self._current_step]})"
            )
            self.insert(
                self.index(next_step_widget) + 1, self._next_step_button
            )


# if we want even more control over our widget, we can use
# magicgui `Container`
class ImageThreshold(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._threshold_slider = create_widget(
            label="Threshold", annotation=float, widget_type="FloatSlider"
        )
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        # use magicgui widgets directly
        self._invert_checkbox = CheckBox(text="Keep pixels below threshold")

        # connect your own callbacks
        self._threshold_slider.changed.connect(self._threshold_im)
        self._invert_checkbox.changed.connect(self._threshold_im)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._image_layer_combo,
                self._threshold_slider,
                self._invert_checkbox,
            ]
        )

    def _threshold_im(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        image = img_as_float(image_layer.data)
        name = image_layer.name + "_thresholded"
        threshold = self._threshold_slider.value
        if self._invert_checkbox.value:
            thresholded = image < threshold
        else:
            thresholded = image > threshold
        if name in self._viewer.layers:
            self._viewer.layers[name].data = thresholded
        else:
            self._viewer.add_labels(thresholded, name=name)
