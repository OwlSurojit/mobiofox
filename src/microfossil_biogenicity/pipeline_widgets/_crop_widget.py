from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import napari.viewer
import numpy as np
from magicgui.widgets import (
    ProgressBar,
    PushButton,
    RangeSlider,
    Widget,
)
from qtpy.QtCore import QObject, QThread, QTimer, Signal
from scipy.signal import find_peaks
from skimage import filters, measure

from ._pipeline_widget import PipelineWidget

# Constants
DEBOUNCE_TIME_MS = 400


class CropWorker(QObject):
    """
    Worker thread to calculate the bounding box of the sample through segmentation and heuristics.
    """

    finished = Signal(tuple)
    progress = Signal()

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.progress_step = 0

    def run(self):
        projection_z = np.max(self.data, axis=0)
        projection_y = np.mean(self.data, axis=1)
        self.progress.emit()

        # Detect xy bounding box through z projection
        threshold_z = filters.threshold_otsu(projection_z.compute())
        self.progress.emit()
        label_img = measure.label(projection_z > threshold_z)
        self.progress.emit()
        regions = measure.regionprops(label_img)
        largest_region = max(regions, key=lambda r: r.area)
        bbox = largest_region.bbox
        self.progress.emit()

        # Detect top of the sample and coating artifact line in y projection
        edges = filters.sobel(projection_y.compute(), axis=0)
        self.progress.emit()
        vertical_edge_strength = edges.mean(axis=1)
        normalized_edge_strength = vertical_edge_strength / np.max(
            np.abs(vertical_edge_strength)
        )
        self.progress.emit()
        pos_peaks, pos_peaks_properties = find_peaks(
            normalized_edge_strength, height=0.01, prominence=0.01, wlen=5
        )
        neg_peaks, _ = find_peaks(
            -normalized_edge_strength, height=0.1, prominence=0.01
        )
        sample_top_z = 0

        # Crop the sample based on detected vertical edges
        if len(pos_peaks) > 1:
            # if there are multiple positive peaks (increase in brightness),
            # the first marks the top of the sample and the last should mark the coating artifact
            sample_top_z = pos_peaks_properties["left_bases"][0]
            sample_bottom_z = pos_peaks_properties["left_bases"][-1]
        elif len(pos_peaks) == 1:
            # If there is only one, check if it is the top or the coating artifact
            # by using the position of the peak as a simple heuristic
            if pos_peaks[0] < self.data.shape[0] / 2:
                sample_top_z = pos_peaks_properties["left_bases"][0]
            else:
                sample_bottom_z = pos_peaks_properties["left_bases"][0]
        elif len(neg_peaks) > 0:
            # If there is no positive peak, but some negative peaks, the last one should be the end of the sample
            sample_bottom_z = neg_peaks[-1]
        else:
            # Otherwise, we assume the whole sample is present
            sample_bottom_z = self.data.shape[0]
        self.progress.emit()

        # Return 3d bounding box as low x, high x, low y, high y, low z, high z
        self.finished.emit(
            (bbox[1], bbox[3], bbox[0], bbox[2], sample_top_z, sample_bottom_z)
        )


class CropWidget(PipelineWidget):
    """
    Widget to crop a fossil scan to contain only the relevant parts of the sample.
    Offers automatic cropping through segmentation and heuristics, as well as manual cropping through sliders.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    input_image_widget : magicgui.Widget, optional
        Picker Widget to choose the input data layer.
        Should be set if part of a pipeline, otherwise it will be created.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(viewer, input_widget, "_cropped", **kwargs)

        self._autocrop_button = PushButton(
            text="Automatically crop around the sample"
        )
        self._autocrop_progress = ProgressBar(
            label="Autocrop progress",
            value=0,
            max=8,
            min=0,
        )
        self._autocrop_progress.hide()
        self._crop_z = RangeSlider(min=0, max=1, value=(0, 1), label="Crop z")
        self._crop_y = RangeSlider(min=0, max=1, value=(0, 1), label="Crop y")
        self._crop_x = RangeSlider(min=0, max=1, value=(0, 1), label="Crop x")

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._crop)

        self._autocrop_button.changed.connect(self._autocrop_around_sample)
        self._crop_z.changed.connect(self._start_debounce_timer)
        self._crop_y.changed.connect(self._start_debounce_timer)
        self._crop_x.changed.connect(self._start_debounce_timer)

        self.extend(
            [
                self._autocrop_button,
                self._autocrop_progress,
                self._crop_z,
                self._crop_y,
                self._crop_x,
            ]
        )

        self._input_changed()

    def _start_background_worker(self):
        self._worker_thread = QThread()
        self._crop_worker = CropWorker(self.input_data)
        self._crop_worker.moveToThread(self._worker_thread)

        # Connect signals
        self._worker_thread.started.connect(self._crop_worker.run)
        self._crop_worker.finished.connect(self._process_background_result)
        self._crop_worker.finished.connect(self._worker_thread.quit)
        self._crop_worker.finished.connect(self._crop_worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._crop_worker.progress.connect(self._step_autocrop_progress)

        self._worker_thread.start()

    def _process_background_result(self, result):
        self._crop_box = result

    def _input_changed(self):
        if not super()._input_changed():
            return
        # Update the crop ranges based on the image data
        self._crop_z.max = self.input_data.shape[0]
        self._crop_z.value = (0, self.input_data.shape[0])
        self._crop_y.max = self.input_data.shape[1]
        self._crop_y.value = (0, self.input_data.shape[1])
        self._crop_x.max = self.input_data.shape[2]
        self._crop_x.value = (0, self.input_data.shape[2])

        self._crop_box = None
        self._autocrop_progress.value = 0
        self._start_background_worker()

    def _start_debounce_timer(self):
        """Start or restart the debounce timer."""
        if getattr(self, "_ignore_slider_callback", False):
            return
        self._debounce_timer.start(DEBOUNCE_TIME_MS)

    def _autocrop_around_sample(self):
        self._autocrop_progress.show()

        if self._crop_box is None:
            QTimer.singleShot(100, self._autocrop_around_sample)
            return

        lox, hix, loy, hiy, loz, hiz = self._crop_box
        self.output_data = self.input_data[loz:hiz, loy:hiy, lox:hix]

        self._step_autocrop_progress()

        # Update sliders
        self._ignore_slider_callback = True
        self._crop_z.value = (loz, hiz)
        self._crop_y.value = (loy, hiy)
        self._crop_x.value = (lox, hix)
        self._ignore_slider_callback = False

        self._autocrop_progress.hide()

    def _step_autocrop_progress(self):
        self._autocrop_progress.value = min(
            self._autocrop_progress.value + 1, self._autocrop_progress.max
        )

    def _crop(self):
        crop_range_z = self._crop_z.value
        crop_range_y = self._crop_y.value
        crop_range_x = self._crop_x.value

        self.output_data = self.input_data[
            crop_range_z[0] : crop_range_z[1],
            crop_range_y[0] : crop_range_y[1],
            crop_range_x[0] : crop_range_x[1],
        ]
