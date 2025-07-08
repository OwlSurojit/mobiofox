import napari
import numpy as np
from magicgui.widgets import ComboBox, FloatSlider, PushButton, SpinBox, Widget

# from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import gaussian_filter1d, median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

from .._utils._util_funcs import debounce
from ._pipeline_widget import PipelineWidget, PipelineWorker


class FilterWorker(PipelineWorker):

    def __init__(
        self, data, mask, filter_type, sigma_for_gaussian=1, niter_for_ad=3
    ):
        super().__init__(data, mask)
        self._filter_type = filter_type
        self._sigma_for_gaussian = sigma_for_gaussian
        self._niter_for_ad = niter_for_ad

    def _gaussian_with_progress(
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
        output = np.zeros(shape=input_data.shape, dtype=input_data.dtype.name)
        blurred_mask = self.mask.astype(np.float32)
        for axis in range(input_data.ndim):
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
            self._increment_progress(50 / input_data.ndim)
            blurred_mask = gaussian_filter1d(
                blurred_mask,
                sigma,
                axis,
                order,
                None,
                mode,
                cval,
                truncate,
                radius=radius,
            )
            self._increment_progress(50 / input_data.ndim)
        with np.errstate(divide="ignore", invalid="ignore"):
            output = (output / blurred_mask).astype(input_data.dtype)
            output[~self.mask] = 0
        return output

    def _anisotropic_diffusion_with_progess(
        self, img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1
    ):
        """
        Helper function to apply anisotropic diffusion with progress updates.
        Adapted from `medpy.filter.smoothing.anisotropic_diffusion`

        Edge-preserving, XD Anisotropic diffusion.

        To achieve the best effects, the image should be scaled to
        values between 0 and 1 beforehand.
        """

        # define conduction gradients functions
        if option == 1:

            def condgradient(delta, spacing):
                return np.exp(-((delta / kappa) ** 2.0)) / float(spacing)

        elif option == 2:

            def condgradient(delta, spacing):
                return 1.0 / (1.0 + (delta / kappa) ** 2.0) / float(spacing)

        elif option == 3:
            kappa_s = kappa * (2**0.5)

            def condgradient(delta, spacing):
                top = (
                    0.5
                    * ((1.0 - (delta / kappa_s) ** 2.0) ** 2.0)
                    / float(spacing)
                )
                return np.where(np.abs(delta) <= kappa_s, top, 0)

        # initialize output array
        out = np.array(img, dtype=np.float32, copy=True)

        # set default voxel spacing if not supplied
        if voxelspacing is None:
            voxelspacing = tuple([1.0] * img.ndim)

        # initialize some internal variables
        deltas = [np.zeros_like(out) for _ in range(out.ndim)]

        iter_progress = 100 / niter

        for _ in range(niter):
            # calculate the diffs
            for i in range(out.ndim):
                slicer = tuple(
                    [
                        slice(None, -1) if j == i else slice(None)
                        for j in range(out.ndim)
                    ]
                )
                deltas[i][tuple(slicer)] = np.diff(out, axis=i)
                self._increment_progress(iter_progress / (3 * out.ndim))

            # update matrices
            matrices = [
                condgradient(delta, spacing) * delta
                for delta, spacing in zip(deltas, voxelspacing, strict=False)
            ]
            self._increment_progress(iter_progress / 6)

            # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
            # pixel. Don't as questions. just do it. trust me.
            for i in range(out.ndim):
                slicer = tuple(
                    [
                        slice(1, None) if j == i else slice(None)
                        for j in range(out.ndim)
                    ]
                )
                matrices[i][tuple(slicer)] = np.diff(matrices[i], axis=i)
                self._increment_progress(iter_progress / (3 * out.ndim))

            # update the image
            out += gamma * (np.sum(matrices, axis=0))

            # Apply mask to prevent bleeding
            # out[~self.mask] = 0

            self._increment_progress(iter_progress / 6)

        return out

    def run(self):
        if self._filter_type == "gaussian":
            output = self._gaussian_with_progress(
                self.data, sigma=self._sigma_for_gaussian
            )
        elif self._filter_type == "median":
            self._set_progress(10)
            output = median_filter(self.data, size=3)
            self._set_progress(90)
        elif self._filter_type == "nlm":
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
        elif self._filter_type == "ad":
            datamin = self.data.min()
            datamax = self.data.max()
            output = self._anisotropic_diffusion_with_progess(
                (self.data - datamin) / (datamax - datamin),
                niter=self._niter_for_ad,
                kappa=50,
                gamma=0.1,
            )
            output = ((output * (datamax - datamin)) + datamin).astype(
                self.data.dtype
            )
            output[~self.mask] = 0
        else:
            output = self.data

        self.finished.emit(output)


class FilterWidget(PipelineWidget):
    """
    Widget to apply various filters and remove artifacts from the scanning process.
    """

    DENOISING_FILTERS = {
        "none": "None",
        "gaussian": "Gaussian",
        "median": "Median",
        "nlm": "Non local means",
        "ad": "Anisotropic Diffusion",
    }

    def __init__(
        self,
        viewer: napari.Viewer,
        input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(viewer, input_widget, "_filtered", **kwargs)

        self._denoising_filter = ComboBox(
            label="Denoising filter",
            choices={
                "choices": self.DENOISING_FILTERS.keys(),
                "key": lambda v: self.DENOISING_FILTERS[v],
            },
            value="gaussian",
        )
        self._denoising_filter.changed.connect(self._filter_changed)

        self._start_button = PushButton(text="Start denoising")
        self._start_button.changed.connect(self._apply_denoising_filter)

        self._gaussian_sigma = FloatSlider(
            label="SD for Gaussian kernel (px)", value=1, min=0.001, max=2
        )
        self._gaussian_sigma.changed.connect(self._restart_if_running)

        self._ad_niter = SpinBox(
            label="AD Iterations", value=3, min=1, max=100, visible=False
        )
        self._ad_niter.changed.connect(self._restart_if_running)

        self._progress_bar.label = "Denoising Progress"

        self.extend(
            [
                self._denoising_filter,
                self._gaussian_sigma,
                self._ad_niter,
                self._start_button,
                self._progress_bar,
            ]
        )

    def input_changed(self):
        if not super().input_changed():
            return

        self.interrupt_worker(self._apply_denoising_filter)

    def _filter_changed(self):
        # Show/hide parameter controls
        if self._denoising_filter.value == "gaussian":
            self._gaussian_sigma.show()
        else:
            self._gaussian_sigma.hide()

        if self._denoising_filter.value == "ad":
            self._ad_niter.show()
        else:
            self._ad_niter.hide()

        if self._denoising_filter.value == "none":
            self._start_button.hide()
            self.output_data = self.input_data
        else:
            self._start_button.show()
            self.interrupt_worker(self._apply_denoising_filter)

    def _apply_denoising_filter(self):
        if self._denoising_filter.value == "none":
            self.output_data = self.input_data
        else:
            print("Denoising")
            self._progress_bar.value = 0
            self._progress_bar.show()
            self.start_background_worker(
                FilterWorker,
                self._denoising_filter.value,
                sigma_for_gaussian=self._gaussian_sigma.value,
                niter_for_ad=self._ad_niter.value,
            )

    @debounce(300)
    def _restart_if_running(self):
        self.interrupt_worker(self._apply_denoising_filter)

    def process_background_result(self, result):
        self.output_data = result
        print("Done")
        self._progress_bar.hide()
