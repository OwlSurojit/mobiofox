import napari
import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    Dialog,
    FloatSpinBox,
    Label,
    PushButton,
    create_widget,
)
from napari.layers import Layer

from ._util_funcs import extract_unit_and_value, get_unit_factor


class MetadataWidget(Dialog):
    def __init__(self, viewer: napari.Viewer, layer: Layer = None):
        super().__init__()
        self._viewer = viewer
        print(layer)
        if layer is None:
            layer = viewer.layers.selection.active
            if layer is None:
                raise UserWarning("No layer selected.")
        self._layer = layer

        self._metadata = getattr(layer, "metadata", {}).copy()

        self._title = Label(value=f"Metadata for {layer.name}")

        unit, pixelsize = (
            extract_unit_and_value(self._metadata["pixelsize"])
            if "pixelsize" in self._metadata
            else ("nm", 10.0)
        )

        def get_adaptive_step(val):
            max_digits = 14
            int_part = int(abs(val))
            magnitude = 1 if int_part == 0 else int(np.log10(int_part)) + 1
            if magnitude >= max_digits:
                return 1.0
            frac_part = abs(val) - int_part
            multiplier = 10 ** (max_digits - magnitude)
            frac_digits = multiplier + int(multiplier * frac_part + 0.5)
            while frac_digits % 10 == 0:
                frac_digits /= 10
            scale = int(np.log10(frac_digits))
            return 10 ** (-scale)

        self._pixel_size = FloatSpinBox(
            min=0, step=get_adaptive_step(pixelsize), value=pixelsize
        )
        self._unit_selection = ComboBox(
            choices=["pm", "nm", "µm", "mm", "m"], value=unit
        )
        self._pixel_size_container = Container(
            layout="horizontal",
            labels=False,
            widgets=[self._pixel_size, self._unit_selection],
            label="Pixel size",
        )

        self._pixel_size_container.native.layout().setContentsMargins(
            0, 0, 0, 0
        )

        self._low_cutoff = FloatSpinBox(
            label="Refracion index low cutoff",
            min=-np.inf,
            max=np.inf,
            step=(
                get_adaptive_step(self._metadata["low_cutoff"])
                if "low_cutoff" in self._metadata
                else 1e-9
            ),
            value=self._metadata.get("low_cutoff", 0.0),
        )
        self._high_cutoff = FloatSpinBox(
            label="Refracion index high cutoff",
            min=-np.inf,
            max=np.inf,
            step=(
                get_adaptive_step(self._metadata["high_cutoff"])
                if "high_cutoff" in self._metadata
                else 1e-9
            ),
            value=self._metadata.get("high_cutoff", 1.0),
        )

        self._factor_edensity = FloatSpinBox(
            label="Factor for electron density",
            step=(
                get_adaptive_step(self._metadata["factor_edensity"])
                if "factor_edensity" in self._metadata
                else 1e-2
            ),
            value=self._metadata.get("factor_edensity", 1.0),
        )

        self.extend(
            [
                self._title,
                self._pixel_size_container,
                self._low_cutoff,
                self._high_cutoff,
                self._factor_edensity,
            ]
        )

    def exec(self):
        if super().exec():
            self._layer.metadata.update(
                {
                    "pixelsize": self._pixel_size.value
                    * get_unit_factor(self._unit_selection.value),
                    "low_cutoff": self._low_cutoff.value,
                    "high_cutoff": self._high_cutoff.value,
                    "factor_edensity": self._factor_edensity.value,
                }
            )


class LayerPickerWithMetadata(Container):

    def __init__(self, viewer: "napari.Viewer", **kwargs):
        super().__init__(layout="horizontal", labels=False, **kwargs)
        self._viewer = viewer

        self.native.layout().setContentsMargins(0, 0, 0, 0)

        self.layer_picker = create_widget(annotation="napari.layers.Image")
        self._open_metadata_button = PushButton(
            text="</>", tooltip="Edit Metadata"
        )
        self._open_metadata_button.changed.connect(self._open_metadata_dialog)
        self._open_metadata_button.native.setMaximumWidth(40)

        self.extend(
            [
                self.layer_picker,
                self._open_metadata_button,
            ]
        )

    def _open_metadata_dialog(self):
        layer = self.layer_picker.value
        if layer is None:
            return

        metadata_widget = MetadataWidget(self._viewer, layer)
        metadata_widget.exec()
