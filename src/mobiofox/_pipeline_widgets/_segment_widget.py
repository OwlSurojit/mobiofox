from functools import partial

import napari
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    IntSlider,
    PushButton,
    SpinBox,
    Widget,
)
from psygnal import Signal
from scipy.stats import norm
from skimage import filters, morphology
from sklearn.cluster import KMeans

from .._utils._histogram import calc_histogram, show_histogram
from .._utils._util_funcs import debounce
from ._pipeline_widget import PipelineWidget, PipelineWorker

INCLUSION_MIN_SIZE = 50


class AutoThresholdWorker(PipelineWorker):
    def __init__(self, data, mask, seeds=None, thresh_counts=1):
        super().__init__(data, mask)
        self._thresholds = None
        self._seeds = seeds
        self._thresh_counts = thresh_counts

    def _threshold_from_seeds(self):
        seeds_int = self._seeds.astype(np.int32)
        seed_intensities = self.data[
            seeds_int[:, 0], seeds_int[:, 1], seeds_int[:, 2]
        ]
        background_intensity = np.median(self.data[self.mask])
        self._increment_progress(20)
        seed_intensities = np.sort([*seed_intensities, background_intensity])
        print("Seed intensities:", seed_intensities)
        # threshs = (seed_intensities[:-1] + seed_intensities[1:]) / 2
        threshs = []
        hist, bin_edges = calc_histogram(self.data[self.mask])
        hist = filters.gaussian(hist, sigma=5, preserve_range=True).astype(
            hist.dtype
        )
        self._increment_progress(10)
        for i in range(len(seed_intensities) - 1):
            lo = bin_edges.searchsorted(seed_intensities[i])
            hi = bin_edges.searchsorted(seed_intensities[i + 1], side="right")
            t_idx = np.argmin(hist[lo:hi]) + lo
            if abs(t_idx - lo) < 5 or abs(t_idx - hi) < 5:
                t = (seed_intensities[i] + seed_intensities[i + 1]) / 2
            else:
                t = bin_edges[t_idx]
            threshs.append(t)
            self._increment_progress(20 / (len(seed_intensities) - 1))
        print("Manual thresholds from seeds:", threshs)
        self.finished.emit(np.array(threshs))

    def _threshold_multiotsu(self):
        self._increment_progress(10)
        thresholds = filters.threshold_multiotsu(
            hist=calc_histogram(self.data[self.mask]),
            classes=self._thresh_counts + 1,
        )
        print("Multiotsu thresholds:", thresholds)
        self._increment_progress(40)
        self.finished.emit(thresholds)

    def run(self):
        if self._seeds is not None:
            print("Using seeds for thresholding")
            self._threshold_from_seeds()
        else:
            self._threshold_multiotsu()
        # threshold_yen = filters.threshold_yen(self.data[self.mask])
        # print("Yen's threshold:", threshold_yen)


class SegmentWorker(PipelineWorker):

    def __init__(
        self,
        data,
        mask,
        segmentation_method,
        thresholds=None,
        matrix_removal_num_std=2,
        after_matrix_removal_method=None,
        kmeans_num_clusters=3,
        seeds=None,
    ):
        super().__init__(data, mask)
        self.segmentation_method = segmentation_method
        self._thresholds = np.sort(thresholds)
        self._matrix_removal_num_std = matrix_removal_num_std
        self._after_matrix_removal_method = after_matrix_removal_method
        self._kmeans_num_clusters = kmeans_num_clusters
        self._seeds = seeds

    def _kmeans(self):
        INTENSITY_IMPORTANCE = 100.0
        intensity = self.data[self.mask]
        min_intensity = intensity.min()
        max_intensity = intensity.max()
        intensity = INTENSITY_IMPORTANCE * (
            (intensity - min_intensity) / (max_intensity - min_intensity)
        )
        z_idx, y_idx, x_idx = np.nonzero(self.mask)
        max_z, max_y, max_x = self.mask.shape
        z_idx = z_idx / max_z
        y_idx = y_idx / max_y
        x_idx = x_idx / max_x
        features = np.column_stack((intensity, z_idx, y_idx, x_idx))
        self._increment_progress(20)
        if self._seeds is not None:
            seeds_int = self._seeds.astype(np.int32)
            seed_intensities = self.data[
                seeds_int[:, 0], seeds_int[:, 1], seeds_int[:, 2]
            ]
            seed_features = np.column_stack(
                (
                    INTENSITY_IMPORTANCE
                    * (
                        (seed_intensities - min_intensity)
                        / (max_intensity - min_intensity)
                    ),
                    self._seeds / np.array(self.mask.shape),
                )
            )
            print("Using seeds for KMeans:\n", seed_features)
            kmeans = KMeans(
                n_clusters=len(seed_features),
                init=seed_features,
                tol=1e-4,
                max_iter=50,
                verbose=1,
                copy_x=False,
            )
        else:
            kmeans = KMeans(
                n_clusters=self._kmeans_num_clusters + 1,
                tol=1e-4,
                max_iter=50,
                verbose=1,
                copy_x=False,
            )
        self._increment_progress(10)
        kmeans.fit(features)
        self._increment_progress(30)
        labels = np.zeros_like(self.data, dtype=np.uint8)
        labels_flat = kmeans.labels_
        self._dominant_label_to_zero(labels_flat)
        self._increment_progress(30)
        labels[self.mask] = labels_flat
        self.finished.emit(labels)

    def _matrix_removal(self):
        projection = np.median(self.data, axis=0)
        self._increment_progress(10)
        mu, std = norm.fit(projection[projection != 0].ravel())
        print(mu, std)
        self._increment_progress(15)
        self.mask = (
            abs(self.data - mu) > self._matrix_removal_num_std * std
        ) & self.mask
        # remove noise
        self.mask = morphology.remove_small_objects(
            self.mask, min_size=INCLUSION_MIN_SIZE
        )
        self._increment_progress(15)
        # noback = np.zeros_like(self.data)
        if self._after_matrix_removal_method == "kmeans":
            self._kmeans()
            return
        elif self._after_matrix_removal_method == "auto_thresholding":
            self._threshold()
            return
        else:
            labels = np.zeros_like(self.data, dtype=np.uint8)
            labels[self.mask] = 1
            self._increment_progress(20)
            self.finished.emit(labels)

    def _dominant_label_to_zero(self, labels):
        l_vals, l_counts = np.unique_counts(labels)
        max_l = l_vals[np.argmax(l_counts)]
        if max_l != 0:
            max_l_mask = labels == max_l
            labels[labels == 0] = max_l
            labels[max_l_mask] = 0

    def _threshold(self):
        if self._thresholds is None:
            raise ValueError("Manual thresholds not provided")
        labels_flat = np.digitize(self.data[self.mask], bins=self._thresholds)
        self._increment_progress(30)
        self._dominant_label_to_zero(labels_flat)
        self._increment_progress(30)
        labels = np.zeros_like(self.data, dtype=np.uint8)
        labels[self.mask] = labels_flat
        self._increment_progress(20)
        self.finished.emit(labels)

    def run(self):
        if self.segmentation_method == "kmeans":
            self._kmeans()
        elif self.segmentation_method == "thresholding":
            self._threshold()
        elif self.segmentation_method == "matrix_removal":
            self._matrix_removal()


class MultiSlider:
    slider_changed = Signal()

    def __init__(
        self,
        parent_widget: Container,
        count_box_label,
        min_count,
        max_count,
        value_count,
        min_slider,
        max_slider,
        visible=False,
    ):

        self._parent_widget = parent_widget
        self._visible = visible
        self._count_box_label = count_box_label
        self._count_selector = SpinBox(
            label=count_box_label,
            value=value_count,
            min=min_count,
            max=max_count,
            step=1,
            visible=self._visible,
        )
        self._count_selector.changed.connect(self._adjust_slider_count)
        self._parent_widget.append(self._count_selector)
        self._last_slider_idx = len(self._parent_widget)

        self._sliders = []
        self._sliders_min = min_slider
        self._sliders_max = max_slider
        self._adjust_slider_count()

    def _adjust_slider_count(self):
        n_sliders = self._count_selector.value
        if n_sliders > len(self._sliders):
            for i in range(len(self._sliders), n_sliders):
                slider = IntSlider(
                    label=f"Threshold {i+1}",
                    min=self._sliders_min,
                    max=self._sliders_max,
                    value=self._sliders_min,
                    step=1,
                    visible=self._visible,
                )
                slider.changed.connect(partial(self._emit_slider_changed, i))
                self._sliders.append(slider)
                self._parent_widget.insert(self._last_slider_idx, slider)
                self._last_slider_idx += 1
        elif n_sliders < len(self._sliders):
            for _ in range(len(self._sliders), n_sliders, -1):
                self._sliders.pop()
                self._parent_widget.pop(self._last_slider_idx - 1)
                self._last_slider_idx -= 1

    def _emit_slider_changed(self, idx):
        print(f"Slider {idx} changed")
        self.slider_changed.emit()

    def update_thresholds(self, thresholds):
        if len(thresholds) > self._count_selector.max:
            self._count_selector.max = len(thresholds)
        self._count_selector.value = len(thresholds)
        for s, t in zip(self._sliders, thresholds, strict=False):
            s.value = t

    def adjust_sliders(self, min_slider, max_slider):
        self._sliders_min = min_slider
        self._sliders_max = max_slider
        for slider in self._sliders:
            slider.min = min_slider
            slider.max = max_slider

    def hide(self):
        self._visible = False
        self._update_visibility()

    def show(self):
        self._visible = True
        self._update_visibility()

    def _update_visibility(self):
        self._count_selector.visible = self._visible
        for slider in self._sliders:
            slider.visible = self._visible

    @property
    def count(self):
        return self._count_selector.value

    @property
    def values(self):
        return [slider.value for slider in self._sliders]


class SegmentWidget(PipelineWidget):

    _SEGMENTATION_METHODS = {
        "kmeans": "KMeans Clustering",
        "thresholding": "Thresholding",
        "matrix_removal": "Matrix removal",
    }

    _AFTER_MATRIX_REMOVAL_METHODS = {
        "none": "None",
        "kmeans": "KMeans Clustering",
        # "thresholding": "Thresholding"
    }

    def __init__(
        self,
        viewer: napari.Viewer,
        input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(
            viewer,
            input_widget,
            output_layer_suffix="_segmented",
            output_layer_is_labels=True,
            **kwargs,
        )
        self._histogram_button = PushButton(text="Show histogram")
        self._histogram_button.changed.connect(self._show_histogram)

        self._segmentation_method = ComboBox(
            label="Segmentation Method",
            choices={
                "choices": self._SEGMENTATION_METHODS.keys(),
                "key": lambda v: self._SEGMENTATION_METHODS[v],
            },
            value="matrix_removal",
        )
        self._segmentation_method.changed.connect(
            self._on_segmentation_type_changed
        )

        self._seeds_button = PushButton(text="Select seed points")
        self._seeds_button.changed.connect(self._select_seeds)

        self._kmeans_num_clusters = SpinBox(
            label="Number of clusters to label", value=3, min=1, max=10, step=1
        )

        self._matrix_removal_num_std = FloatSlider(
            label="SDs to remove around matrix mean",
            value=1.5,
            min=0.5,
            max=3,
            step=0.001,
        )
        self._after_matrix_removal = ComboBox(
            label="Method after matrix removal",
            choices={
                "choices": self._AFTER_MATRIX_REMOVAL_METHODS.keys(),
                "key": lambda v: self._AFTER_MATRIX_REMOVAL_METHODS[v],
            },
            value="kmeans",
        )
        self._after_matrix_removal.changed.connect(
            self._on_after_matrix_removal_changed
        )

        self.extend(
            [
                self._histogram_button,
                self._seeds_button,
                self._segmentation_method,
                self._matrix_removal_num_std,
                self._after_matrix_removal,
                self._kmeans_num_clusters,
            ]
        )

        if self.input_data is not None:
            input_min = self.input_data[self.input_mask].min()
            input_max = self.input_data[self.input_mask].max()
        else:
            input_min = 0
            input_max = 2**16 - 1

        self._manual_thresholds_widget = MultiSlider(
            parent_widget=self,
            count_box_label="Number of thresholds",
            min_count=1,
            max_count=10,
            value_count=2,
            min_slider=input_min,
            max_slider=input_max,
            visible=False,
        )
        self._manual_thresholds_widget.slider_changed.connect(
            self._manual_threshs_changed
        )

        self._start_button = PushButton(text="Start segmentation")
        self._start_button.changed.connect(self._start_segmentation)

        self._auto_threshold_button = PushButton(
            text="Determine thresholds automatically", visible=False
        )
        self._auto_threshold_button.changed.connect(
            self._start_auto_thresholding
        )

        self._progress_bar.label = "Segmentation Progress"

        self.extend(
            [
                self._auto_threshold_button,
                self._start_button,
                self._progress_bar,
            ]
        )

    def input_changed(self):
        if not super().input_changed():
            return False

        self._manual_thresholds_widget.adjust_sliders(
            self.input_data.min(), self.input_data.max()
        )

        if self.visible:
            self.interrupt_worker(self._start_segmentation)

    def _show_histogram(self):
        if self.input_data is None:
            return
        show_histogram(
            self.input_data[self.input_mask],
            layer_name=self._input_widget.value.name,
        )

    def _on_segmentation_type_changed(self):
        self._matrix_removal_num_std.hide()
        self._manual_thresholds_widget.hide()
        self._kmeans_num_clusters.hide()
        self._after_matrix_removal.hide()
        self._auto_threshold_button.hide()

        if self._segmentation_method.value == "thresholding":
            self._manual_thresholds_widget.show()
            self._auto_threshold_button.show()
        elif self._segmentation_method.value == "matrix_removal":
            self._matrix_removal_num_std.show()
            self._after_matrix_removal.show()
            self._on_after_matrix_removal_changed()
        elif self._segmentation_method.value == "kmeans":
            self._kmeans_num_clusters.show()

        self.interrupt_worker(self._start_segmentation)

    def _on_after_matrix_removal_changed(self):
        self._kmeans_num_clusters.hide()
        if self._after_matrix_removal.value == "kmeans":
            self._kmeans_num_clusters.show()
        elif self._after_matrix_removal.value == "auto_thresholding":
            pass

    def _select_seeds(self):
        if self.input_data is None:
            return

        seeds_layer_name = f"{self._input_widget.value.name}_seeds"
        if seeds_layer_name in self._viewer.layers:
            self._seeds_layer = self._viewer.layers[seeds_layer_name]
        else:
            self._seeds_layer = self._viewer.add_points(
                ndim=3,
                name=seeds_layer_name,
                size=5,
                face_color="red",
                border_color="#ff9bdc",
            )

        if self._viewer.dims.ndisplay != 2:
            self._viewer.dims.ndisplay = 2
        self._input_widget.value.visible = True
        self._viewer.layers.selection = [self._seeds_layer]

        self._auto_threshold_button.text = (
            "Determine thresholds based on seeds"
        )

        self._seeds_layer.mode = "add"
        self._seeds_layer.events.data.connect(self._seeds_changed)
        self._viewer.layers.events.removed.connect(self._layer_removed)

    def _seeds_changed(self):
        if not hasattr(self, "_seeds_layer"):
            return
        print("Seeds changed:", self._seeds_layer.data)

        seg_meth = (
            self._after_matrix_removal.value
            if self._segmentation_method.value == "matrix_removal"
            else self._segmentation_method.value
        )

        if seg_meth == "kmeans":
            self._kmeans_num_clusters.value = max(
                1, len(self._seeds_layer.data)
            )

    def _layer_removed(self, event):
        if event.value == self._seeds_layer:
            print("Seeds layer removed")
            self._seeds_layer.events.data.disconnect(self._seeds_changed)
            self._seeds_layer = None
            self._auto_threshold_button.text = (
                "Determine thresholds automatically"
            )

    @debounce(300)
    def _manual_threshs_changed(self):
        if not self.interrupt_worker(self._start_segmentation):
            self._start_segmentation()

    def _start_auto_thresholding(self):
        self._progress_bar.value = 0
        self._progress_bar.show()
        self.start_background_worker(
            AutoThresholdWorker,
            result_callback=self._process_auto_threshs,
            seeds=(
                self._seeds_layer.data
                if getattr(self, "_seeds_layer", None) is not None
                else None
            ),
            thresh_counts=self._manual_thresholds_widget.count,
        )

    def _process_auto_threshs(self, thresholds):
        self._manual_thresholds_widget.update_thresholds(thresholds)

    def _start_segmentation(self):
        self._progress_bar.value = 0
        self._progress_bar.show()
        self.start_background_worker(
            SegmentWorker,
            segmentation_method=self._segmentation_method.value,
            thresholds=self._manual_thresholds_widget.values,
            matrix_removal_num_std=self._matrix_removal_num_std.value,
            after_matrix_removal_method=self._after_matrix_removal.value,
            kmeans_num_clusters=self._kmeans_num_clusters.value,
            seeds=(
                self._seeds_layer.data
                if getattr(self, "_seeds_layer", None) is not None
                else None
            ),
        )

    def process_background_result(self, result):
        self.output_data = result
        self._progress_bar.hide()
