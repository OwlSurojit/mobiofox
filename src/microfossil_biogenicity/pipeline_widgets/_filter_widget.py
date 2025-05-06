from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np
from magicgui.widgets import ComboBox, FloatSlider, ProgressBar, Widget
from medpy.filter.smoothing import anisotropic_diffusion
from qtpy.QtCore import Signal
from scipy.ndimage import gaussian_filter1d
from skimage.restoration import denoise_nl_means, estimate_sigma

from ._pipeline_widget import PipelineWidget, PipelineWorker


class FilterWorker(PipelineWorker):
    interrupted = Signal()

    def __init__(self, data, filter_type, sigma_for_gaussian=1):
        super().__init__(data)
        self._filter_type = filter_type
        self.sigma_for_gaussian = sigma_for_gaussian
        self._running = True

    def stop(self):
        self._running = False

    def gaussian_with_progress(
        self,
        input_data,
        sigma=1,
        order=0,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        *,
        radius=None,
    ):
        """
        Helper function to apply gaussian filter with progress updates.
        Adapted from `scipy.ndimage.gaussian_filter
        """
        # output = np.zeros_like(input)
        output = np.zeros(shape=input_data.shape, dtype=input_data.dtype.name)
        for axis in range(input_data.ndim):
            if not self._running:
                break
            gaussian_filter1d(
                input_data,
                sigma,
                axis,
                order,
                output,
                mode,
                cval,
                truncate,
                radius=radius,
            )
            input_data = output
            self.progress.emit()
        return output

    def run(self):
        if self._filter_type == "Gaussian":
            output = self.gaussian_with_progress(
                self.data, sigma=self.sigma_for_gaussian
            )
        elif self._filter_type == "Non local means":
            sigma_est = estimate_sigma(self.data, average_sigmas=True)
            print(f"Estimated noise standard deviation = {sigma_est}")
            self.data = self.data.compute()
            output = denoise_nl_means(
                self.data,
                h=0.8 * sigma_est,
                sigma=sigma_est,
                fast_mode=True,
                patch_size=5,
                patch_distance=1,
                preserve_range=True,
            )
        elif self._filter_type == "Anisotropic Diffusion":
            output = anisotropic_diffusion(
                self.data / (2**16 - 1), niter=1, kappa=50, gamma=0.1
            ) * (2**16 - 1)
        else:
            output = self.data

        if not self._running:
            self.interrupted.emit()
            output = self.data
        self.finished.emit(output)


class FilterWidget(PipelineWidget):
    """
    Widget to apply various filters and remove artifacts from the scanning process.
    """

    _BACKGROUND_WORKER_TYPE = FilterWorker
    DENOISING_FILTERS = [
        "None",
        "Gaussian",
        "Non local means",
        "Anisotropic Diffusion",
    ]

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(viewer, input_widget, "_filtered", **kwargs)

        self._denoising_filter = ComboBox(
            label="Denoising filter",
            choices=self.DENOISING_FILTERS,
            value="None",
        )
        self._denoising_filter.changed.connect(self._apply_denoising_filter)

        self._gaussian_sigma = FloatSlider(
            label="SD for Gaussian kernel", value=1, min=0.001, max=2
        )
        self._gaussian_sigma.changed.connect(self._adjust_gaussian_sigma)
        self._gaussian_sigma.hide()

        self._denoising_progress = ProgressBar(
            label="Denoising Progress", value=0, min=0, max=4
        )
        self._denoising_progress.hide()

        self.extend(
            [
                self._denoising_filter,
                self._gaussian_sigma,
                self._denoising_progress,
            ]
        )

    def _input_changed(self):
        if not super()._input_changed():
            return

        if self.visible:
            self._apply_denoising_filter()

    def _apply_denoising_filter(self):
        # Show/hide parameter controls
        if self._denoising_filter.value == "Gaussian":
            self._gaussian_sigma.show()
        else:
            self._gaussian_sigma.hide()

        if self._denoising_filter.value == "None":
            self.output_data = self.input_data
        else:
            print("Starting")
            self._denoising_progress.value = 0
            self._denoising_progress.show()
            self._start_background_worker(
                self._denoising_filter.value,
                sigma_for_gaussian=self._gaussian_sigma.value,
            )

    @PipelineWidget.debounce(300)
    def _adjust_gaussian_sigma(self):
        try:
            self._worker.finished.connect(self._apply_denoising_filter)
            self._worker.stop()
        except RuntimeError:
            self._apply_denoising_filter()

    def _process_background_result(self, result):
        self.output_data = result
        print("Done")
        self._denoising_progress.hide()

    def _handle_progress(self):
        self._denoising_progress.value = min(
            self._denoising_progress.value + 1, self._denoising_progress.max
        )
