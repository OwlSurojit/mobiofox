from collections.abc import Callable

import dask.array as da
import napari
import numpy as np
import psygnal
import sip
from magicgui.widgets import Container, ProgressBar, Widget, create_widget
from qtpy.QtCore import QObject, QThread, Signal


class PipelineWorker(QObject):
    """
    Base class for background workers solving computationally intensive tasks for the respective widgets.
    """

    finished = Signal(object)
    progress = Signal(float)
    interrupted = Signal()

    class PipelineWorkerInterrupted(Exception):
        pass

    def __init__(self, data, mask):
        super().__init__()
        if isinstance(data, da.Array):
            self.data = data.compute()
        else:
            self.data = data
        if isinstance(mask, da.Array):
            self.mask = mask.compute()
        else:
            self.mask = mask
        self._running = True
        self._progress_value = 0

    def stop(self):
        self._running = False

    def _increment_progress(self, increment=5):
        self._progress_value += increment
        self._set_progress()

    def _set_progress(self, value=None):
        """
        Emit progress signal and check if the worker is still running.
        """
        if not self._running:
            raise self.PipelineWorkerInterrupted
        if value is None:
            value = self._progress_value
        self._progress_value = min(value, 100)
        self.progress.emit(self._progress_value)

    def start(self):
        try:
            print("Running worker...")
            self.run()
        except self.PipelineWorkerInterrupted:
            print("Worker interrupted.")
            self.interrupted.emit()

    def run(self):
        print(
            "Running base worker. You should probably pass worker_type to _start_background_worker."
        )


class PipelineWidget(Container):
    """
    Base class for all widgets that can act as part of the microfossil biogenicity critera extraction pipeline.
    """

    output_changed = psygnal.Signal(
        object, description="Emitted when the pipeline output changes."
    )

    def __init__(
        self,
        viewer: napari.Viewer,
        input_widget: Widget | None = None,
        output_layer_suffix: str = "_",
        output_layer_is_labels=False,
        input_layer_is_labels=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._viewer = viewer
        if input_widget is not None:
            self._input_widget = input_widget
        else:
            self._input_widget = create_widget(
                label="Input Data",
                annotation=(
                    "napari.layers.Labels"
                    if input_layer_is_labels
                    else "napari.layers.Image"
                ),
            )
            self.append(self._input_widget)

        self._progress_bar = ProgressBar(
            label="Progress", min=0, max=100, value=0
        )
        self._progress_bar.hide()

        self.input_data = None
        self.input_mask = None
        self._output_layer_suffix = output_layer_suffix
        self._output_layer_is_labels = output_layer_is_labels
        try:
            self._input_widget.output_changed.connect(self.input_changed)
        except AttributeError:
            self._input_widget.changed.connect(self.input_changed)
        self._add_output_layer()

    def _add_output_layer(self):
        image_layer = self._input_widget.value
        if image_layer is None:
            return False
        self.input_data = image_layer.data

        if hasattr(self._input_widget, "output_mask"):
            self.input_mask = self._input_widget.output_mask
        else:
            self.input_mask = self.input_data != 0
        self.output_mask = self.input_mask

        self._output_layer_name = image_layer.name + self._output_layer_suffix
        if self._output_layer_name not in self._viewer.layers:
            if self._output_layer_is_labels:
                self._viewer.add_labels(
                    np.zeros_like(self.input_data, dtype=np.uint8),
                    name=self._output_layer_name,
                )
            else:
                self._viewer.add_image(
                    self.input_data, name=self._output_layer_name
                )
                image_layer.visible = False
        else:
            if not self._output_layer_is_labels:
                self._viewer.layers[self._output_layer_name].data = (
                    self.input_data
                )
        self._output_layer = self._viewer.layers[self._output_layer_name]
        return True

    def input_changed(self):
        return self._add_output_layer()

    @property
    def masked_input(self):
        """
        Return the input data masked by the input mask.
        """
        if self.input_mask is not None:
            return self.input_data[self.input_mask]
        return self.input_data

    @property
    def output_data(self):
        return self.value.data

    @output_data.setter
    def output_data(self, value):
        if self.value is None:
            return
        self.value.data = value
        self.output_changed.emit(self.value)

    # Alias for the output layer
    @property
    def value(self):
        return getattr(self, "_output_layer", None)

    @value.setter
    def value(self, value):
        self._output_layer = value

    def start_background_worker(
        self,
        worker_type: type[PipelineWorker] = PipelineWorker,
        *args,
        result_callback=None,
        progress_callback=None,
        **kwargs,
    ):
        if not hasattr(self, "_worker_thread"):
            self._worker_thread = QThread()

        if hasattr(self, "_worker") and not sip.isdeleted(self._worker):
            self._worker.disconnect()

        self._worker = worker_type(
            self.input_data, self.input_mask, *args, **kwargs
        )
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._worker_thread.started.connect(self._worker.start)
        self._worker.finished.connect(
            self.process_background_result
            if result_callback is None
            else result_callback
        )
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.interrupted.connect(self._worker_thread.quit)
        self._worker.interrupted.connect(self._worker.deleteLater)

        # self._worker.finished.connect(lambda: setattr(self, '_worker', None))
        # self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        # self._worker_thread.finished.connect(lambda: setattr(self, '_worker_thread', None))
        self._worker.progress.connect(
            self._handle_progress
            if progress_callback is None
            else progress_callback
        )

        if self._worker_thread.isRunning():
            self._worker_thread.started.emit()
        else:
            self._worker_thread.start()

    def interrupt_worker(self, run_after: Callable = None):
        """
        Interrupt the worker and optionally run a function after interruption.
        """
        if hasattr(self, "_worker") and not (
            sip.isdeleted(self._worker) or self._worker_thread.isFinished()
        ):
            if run_after is not None:
                self._worker.interrupted.connect(run_after)
            self._worker.stop()
            self._progress_bar.value = 0
            return True
        return False

    def process_background_result(self, result):
        pass

    def _handle_progress(self, value):
        self._progress_bar.value = value
