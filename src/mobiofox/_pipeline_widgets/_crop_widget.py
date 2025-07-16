import dask.array as da
import napari
import numpy as np
import sip
from magicgui.widgets import ComboBox, PushButton, RangeSlider, SpinBox, Widget
from qtpy.QtCore import QTimer, Signal
from scipy.signal import find_peaks
from skimage import draw, filters, measure, morphology
from sklearn.linear_model import TheilSenRegressor

from .._utils._util_funcs import debounce
from ._pipeline_widget import PipelineWidget, PipelineWorker


class CropWorker(PipelineWorker):
    """
    Worker thread to calculate the bounding box of the sample through segmentation and heuristics.
    """

    finished = Signal(object)
    progress = Signal(float)

    def __init__(self, data, mask, crop_type):
        super().__init__(data, mask)
        self._crop_type = crop_type
        self._computed_frustum = None
        self._frustum_params = None
        self._bbox = None
        self._sample_top_z = None
        self._sample_bottom_z = None

    def _detect_top_bottom(self):
        projection_y = np.mean(self.data, axis=1)
        self._increment_progress(5)

        # Detect top of the sample and coating artifact line in y projection
        edges = filters.sobel(projection_y, axis=0)
        self._increment_progress(5)
        vertical_edge_strength = edges.mean(axis=1)
        normalized_edge_strength = vertical_edge_strength / np.max(
            np.abs(vertical_edge_strength)
        )
        self._increment_progress(5)
        pos_peaks, pos_peaks_properties = find_peaks(
            normalized_edge_strength, height=0.05, prominence=0.01, wlen=5
        )
        neg_peaks, _ = find_peaks(
            -normalized_edge_strength, height=0.1, prominence=0.01
        )
        self._sample_top_z = 0
        self._sample_bottom_z = self.data.shape[0]

        # Crop the sample based on detected vertical edges
        if len(pos_peaks) > 1:
            # if there are multiple positive peaks (increase in brightness),
            # the first marks the top of the sample and the last should mark the coating artifact
            self._sample_top_z = pos_peaks_properties["left_bases"][0]
            self._sample_bottom_z = pos_peaks_properties["left_bases"][-1]
        elif len(pos_peaks) == 1:
            # If there is only one, check if it is the top or the coating artifact
            # by using the position of the peak as a simple heuristic
            if pos_peaks[0] < self.data.shape[0] / 2:
                self._sample_top_z = pos_peaks_properties["left_bases"][0]
            else:
                self._sample_bottom_z = pos_peaks_properties["left_bases"][0]
        elif len(neg_peaks) > 0:
            # If there is no positive peak, but some negative peaks, the last one should be the end of the sample
            self._sample_bottom_z = neg_peaks[-1]
        # Otherwise, we assume the whole sample is present
        self._increment_progress(10)

    def _detect_xy_bounds(self):
        # Detect xy bounding box through z projection
        projection_z = np.max(self.data, axis=0)
        self._increment_progress(15)
        threshold_z = filters.threshold_otsu(projection_z)
        self._increment_progress(15)
        label_img = measure.label(projection_z > threshold_z)
        self._increment_progress(20)
        regions = measure.regionprops(label_img)
        largest_region = max(regions, key=lambda r: r.area)
        self._bbox = largest_region.bbox
        self._increment_progress(20)

    def _detect_frustum(self):
        sample_height = self._sample_bottom_z - self._sample_top_z

        # Fitting a frustum (conical) to the data
        thresh = filters.threshold_mean(self.data)
        self._increment_progress(5)
        connected = morphology.remove_small_holes(
            self.data > thresh, area_threshold=5000
        ).astype(
            np.uint8
        )  # TODO make this relative to the resolution
        self._increment_progress(15)
        centers = []
        radii = []
        orientations = []

        self._computed_frustum = np.zeros_like(self.data, dtype=bool)
        self._increment_progress(5)

        for i in range(self._sample_top_z, self._sample_bottom_z):
            regions = measure.regionprops(connected[i])
            if len(regions) > 1:
                disk_region = max(regions, key=lambda r: r.area)
            elif len(regions) == 1:
                disk_region = regions[0]
            else:
                centers.append(centers[-1] if centers else (0, 0))
                radii.append(radii[-1] if radii else (0, 0))
                orientations.append(orientations[-1] if orientations else 0)
            centers.append(disk_region.centroid)
            radii.append(
                (
                    disk_region.axis_major_length / 2,
                    disk_region.axis_minor_length / 2,
                )
            )
            orientations.append(disk_region.orientation)
            self._increment_progress(20 / sample_height)

        # Fit lines to center, orientation, and radii, using TheilSen to disregard outliers
        X = np.arange(sample_height).reshape(-1, 1)
        radii = np.array(radii)
        centers = np.array(centers)
        orientations = np.array(orientations)

        regressor_center_x = TheilSenRegressor()
        regressor_center_y = TheilSenRegressor()
        regressor_center_x.fit(X, centers[:, 0])
        regressor_center_y.fit(X, centers[:, 1])
        fitted_center_x = regressor_center_x.predict(X)
        fitted_center_y = regressor_center_y.predict(X)

        regressor_orientation = TheilSenRegressor()
        regressor_orientation.fit(X, orientations)
        fitted_orientation = regressor_orientation.predict(X)

        regressor_major_rad = TheilSenRegressor()
        regressor_minor_rad = TheilSenRegressor()
        regressor_major_rad.fit(X, radii[:, 0])
        regressor_minor_rad.fit(X, radii[:, 1])
        fitted_major_radius = regressor_major_rad.predict(X)
        fitted_minor_radius = regressor_minor_rad.predict(X)
        self._increment_progress(10)

        """for i in range(self._sample_top_z, self._sample_bottom_z):
            fitted_disk = draw.ellipse(fitted_center_x[i], fitted_center_y[i], fitted_major_radius[i], fitted_minor_radius[i], shape=self._computed_frustum[i].shape, rotation=fitted_orientation[i])
            self._computed_frustum[i, *fitted_disk] = True
            self._increment_progress(20/sample_height)"""

        self._frustum_params = (
            fitted_center_x,
            fitted_center_y,
            fitted_major_radius,
            fitted_minor_radius,
            fitted_orientation,
        )

    def run(self):
        if self._sample_top_z is None or self._sample_bottom_z is None:
            self._detect_top_bottom()

        if self._crop_type == "box":
            if self._bbox is None:
                self._detect_xy_bounds()
            loy, lox, hiy, hix = self._bbox
            self.finished.emit(
                (lox, hix, loy, hiy, self._sample_top_z, self._sample_bottom_z)
            )

        elif self._crop_type == "frustum":
            if self._frustum_params is None:
                self._detect_frustum()
            self.finished.emit(
                (
                    self._sample_top_z,
                    self._sample_bottom_z,
                    *self._frustum_params,
                )
            )


class CropWidget(PipelineWidget):
    """
    Widget to crop a fossil scan to contain only the relevant parts of the sample.
    Offers automatic cropping through segmentation and heuristics, as well as manual cropping through sliders.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    input_image_widget : magicgui.Widget, optional
        Picker Widget to choose the input data layer.
        Should be set if part of a pipeline, otherwise it will be created.
    """

    _CROP_TYPES = {
        "frustum": "pillar (cylinder/frustum of elliptic cone)",
        "box": "box",
    }

    def __init__(
        self,
        viewer: napari.Viewer,
        input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(
            viewer, input_widget, output_layer_suffix="_cropped", **kwargs
        )

        self._crop_type = ComboBox(
            label="Sample shape",
            choices={
                "choices": self._CROP_TYPES.keys(),
                "key": lambda v: self._CROP_TYPES[v],
            },
            value="frustum",
        )
        self._crop_type.changed.connect(self._on_crop_type_changed)

        self._autocrop_button = PushButton(
            text="Automatically crop around the sample"
        )
        self._progress_bar.label = "Autocrop progress"
        self._crop_z = RangeSlider(
            min=0, max=1, value=(0, 1), label="Crop z", visible=True
        )
        self._crop_y = RangeSlider(
            min=0, max=1, value=(0, 1), label="Crop y", visible=False
        )
        self._crop_x = RangeSlider(
            min=0, max=1, value=(0, 1), label="Crop x", visible=False
        )

        self._radius_margin = SpinBox(
            label="Radius margin (px)",
            value=5,
            min=0,
            max=50,
            step=1,
            visible=True,
        )

        self.extend(
            [
                self._crop_type,
                self._autocrop_button,
                self._progress_bar,
                self._crop_z,
                self._crop_y,
                self._crop_x,
                self._radius_margin,
            ]
        )

        self.input_changed()

        self._autocrop_button.changed.connect(self._autocrop_around_sample)
        self._crop_z.changed.connect(self._on_crop_sliders_changed)
        self._crop_y.changed.connect(self._on_crop_sliders_changed)
        self._crop_x.changed.connect(self._on_crop_sliders_changed)
        self._radius_margin.changed.connect(self._on_radius_margin_changed)

    def input_changed(self):
        if not super().input_changed():
            return
        # Update the crop ranges based on the image data
        self._crop_z.max = self.input_data.shape[0]
        self._crop_z.value = (0, self.input_data.shape[0])
        self._crop_y.max = self.input_data.shape[1]
        self._crop_y.value = (0, self.input_data.shape[1])
        self._crop_x.max = self.input_data.shape[2]
        self._crop_x.value = (0, self.input_data.shape[2])

        self._crop_box = None
        self.output_mask = None
        self._frustum_params = None
        if not self.interrupt_worker(self._compute_crop_params):
            self._compute_crop_params()

    def _compute_crop_params(self):
        print("Computing crop params")
        self._progress_bar.value = 0
        self.start_background_worker(
            CropWorker, crop_type=self._crop_type.value
        )

    def _on_crop_type_changed(self):
        print(self._crop_type.value)

        if sip.isdeleted(self._worker) or self._worker_thread.isFinished():
            self._compute_crop_params()
        else:
            self._worker.interrupted.connect(self._compute_crop_params)
            self._worker.stop()
            self._progress_bar.value = 0

        if self._crop_type.value == "frustum":
            self._crop_z.hide()
            self._crop_y.hide()
            self._crop_x.hide()
            self._radius_margin.show()
        elif self._crop_type.value == "box":
            self._crop_z.show()
            self._crop_y.show()
            self._crop_x.show()
            self._radius_margin.hide()

    @debounce(400)
    def _on_radius_margin_changed(self):
        if getattr(self, "_frustum_params", None) is None:
            return
        self._compute_mask_from_frustum(self._frustum_params)
        self._apply_mask()

    @debounce(400)
    def _on_crop_sliders_changed(self):
        if getattr(self, "_ignore_slider_callback", False):
            self._ignore_slider_callback = False
            return
        if (
            self._crop_type.value == "frustum"
            and self._frustum_params is not None
        ):
            self._frustum_params = (
                *self._crop_z.value,
                *self._frustum_params[2:],
            )
            self._compute_mask_from_frustum(self._frustum_params)
        else:
            self._compute_mask_from_bbox(
                (*self._crop_x.value, *self._crop_y.value, *self._crop_z.value)
            )
        self._apply_mask()

    def _apply_mask(self):
        # mask_layer_name = self._output_layer_name + "_mask"
        # if mask_layer_name in self._viewer.layers:
        #     self._viewer.layers[mask_layer_name].data = self.output_mask
        # else:
        #     self._viewer.add_labels(self.output_mask, name=mask_layer_name)
        self.output_data = da.where(self.output_mask, self.input_data, 0)

    def _compute_mask_from_bbox(self, bbox):
        lox, hix, loy, hiy, loz, hiz = bbox
        self.output_mask = np.zeros_like(self.input_data, dtype=bool)
        self.output_mask[loz:hiz, loy:hiy, lox:hix] = True

    def _compute_mask_from_frustum(self, frustum_params):
        (
            top_z,
            bottom_z,
            fitted_center_x,
            fitted_center_y,
            fitted_major_radius,
            fitted_minor_radius,
            fitted_orientation,
        ) = frustum_params
        self.output_mask = np.array(np.zeros_like(self.input_data, dtype=bool))
        self._handle_progress(85)
        sample_height = bottom_z - top_z
        shape = self.output_mask[0].shape
        for i in range(sample_height):
            slice_idx = i + top_z - self._crop_z.min
            ellipse_rr, ellipse_cc = draw.ellipse(
                fitted_center_x[slice_idx],
                fitted_center_y[slice_idx],
                fitted_major_radius[slice_idx] - self._radius_margin.value,
                fitted_minor_radius[slice_idx] - self._radius_margin.value,
                shape=shape,
                rotation=fitted_orientation[slice_idx],
            )
            self.output_mask[i + top_z, ellipse_rr, ellipse_cc] = True
            self._handle_progress(85 + 10 * i / sample_height)

    def process_background_result(self, result):
        if self._crop_type.value == "frustum":
            self._frustum_params = result
            self._compute_mask_from_frustum(result)
        elif self._crop_type.value == "box":
            self._crop_box = result
            self._compute_mask_from_bbox(result)

    def _autocrop_around_sample(self):
        self._progress_bar.show()

        if self.output_mask is None:
            QTimer.singleShot(100, self._autocrop_around_sample)
            return

        self._apply_mask()

        if self._crop_type.value == "box" and self._crop_box is not None:
            # Update sliders
            lox, hix, loy, hiy, loz, hiz = self._crop_box
            self._ignore_slider_callback = True
            self._crop_z.value = (loz, hiz)
            self._crop_y.value = (loy, hiy)
            self._crop_x.value = (lox, hix)
        elif (
            self._crop_type.value == "frustum"
            and self._frustum_params is not None
        ):
            loz, hiz = self._frustum_params[:2]
            self._ignore_slider_callback = True
            self._crop_z.range = (loz, hiz)
            self._crop_z.value = (loz, hiz)

        self._progress_bar.hide()
