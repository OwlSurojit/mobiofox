from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from functools import wraps

import psygnal
from magicgui.widgets import Container, Widget, create_widget
from qtpy.QtCore import QObject, QThread, QTimer, Signal


class PipelineWorker(QObject):
    """
    Base class for background workers solving computationally intensive tasks for the respective widgets.
    """

    finished = Signal(object)
    progress = Signal()

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        pass


class PipelineWidget(Container):
    """
    Base class for all widgets that can act as part of the microfossil biogenicity critera extraction pipeline.
    """

    _BACKGROUND_WORKER_TYPE = PipelineWorker
    output_changed = psygnal.Signal(
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
        try:
            self._input_widget.output_changed.connect(self._input_changed)
        except AttributeError:
            self._input_widget.changed.connect(self._input_changed)

    def _add_output_layer(self):
        image_layer = self._input_widget.value
        if image_layer is None:
            return False
        self.input_data = image_layer.data

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

    @staticmethod
    def debounce(wait_time_ms):
        """
        A decorator to debounce a method using a QTimer.

        Parameters:
            wait_time_ms (int): The debounce time in milliseconds.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if not hasattr(self, "_debounce_timers"):
                    self._debounce_timers = {}
                if func not in self._debounce_timers:
                    self._debounce_timers[func] = QTimer()
                    self._debounce_timers[func].setSingleShot(True)
                    self._debounce_timers[func].timeout.connect(
                        lambda: func(self, *args, **kwargs)
                    )
                self._debounce_timers[func].start(wait_time_ms)

            return wrapper

        return decorator

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

    def _start_background_worker(self, *args, timeout=None, **kwargs):
        if not hasattr(self, "_worker_thread"):
            self._worker_thread = QThread()
        self._worker = self._BACKGROUND_WORKER_TYPE(
            self.input_data, *args, **kwargs
        )
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._process_background_result)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        # self._worker.finished.connect(lambda: setattr(self, '_worker', None))
        # self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        # self._worker_thread.finished.connect(lambda: setattr(self, '_worker_thread', None))
        self._worker.progress.connect(self._handle_progress)

        if timeout is not None:
            # Add a timer to monitor the worker
            self._timeout_timer = QTimer()
            self._timeout_timer.setSingleShot(True)
            self._timeout_timer.timeout.connect(self._cancel_worker)
            self._worker.finished.connect(self._timeout_timer.stop)
            self._timeout_timer.start(timeout * 1000)

        self._worker_thread.start()

    def _cancel_worker(self):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            print("Worker terminated due to timeout.")

    def _process_background_result(self, result):
        pass

    def _handle_progress(self):
        pass
