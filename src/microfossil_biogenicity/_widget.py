from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    create_widget,
)
from skimage.util import img_as_float

# Import local modules
from .pipeline_widgets import CropWidget, FilterWidget


class MorphometryPipelineWidget(Container):
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

        # Step 1: Crop
        self.crop_widget = CropWidget(viewer, self._input_image_picker)
        # Step 2: Filter & remove artifacts
        self.filter_widget = FilterWidget(viewer, self.crop_widget)
        # Step 3: Segment inclusions
        # Step 4: Extract morphometry features

        self.extend(
            [
                self._input_image_picker,
                Label(label="Step 1:", value="Crop"),
                self.crop_widget,
                Label(label="Step 2:", value="Filter & remove artifacts"),
                self.filter_widget,
            ]
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
