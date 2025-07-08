import warnings
from math import sqrt

import dask.array as da
import napari
import numpy as np
import pandas as pd
import sip
from magicgui.widgets import PushButton, RadioButtons, Widget
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from skimage import measure, morphology

from .._result_widget import ResultWidget
from .._utils._metadata_widget import LayerPickerWithMetadata
from .._utils._util_funcs import debounce, extract_unit_and_value
from ._pipeline_widget import PipelineWidget, PipelineWorker


class MorphometryWorker(PipelineWorker):
    def __init__(
        self, data, mask, intensity_data, label_choice="All", metadata=None
    ):
        super().__init__(data, mask)
        if isinstance(intensity_data, da.Array):
            self._intensity_data = intensity_data.compute()
        else:
            self._intensity_data = intensity_data
        self._label_choice = label_choice
        self._metadata = metadata if metadata is not None else {}

    def fit_spheres(self, around_label):
        dt = distance_transform_edt(around_label)
        max_radius = -1
        radii, centers = [], []
        while True:
            center = np.unravel_index(np.argmax(dt), dt.shape)
            radius = dt[center]
            if radius > max_radius:
                max_radius = radius
            elif radius < max_radius / 2:
                break

            radii.append(radius)
            centers.append(center)

            zz, yy, xx = np.ogrid[: dt.shape[0], : dt.shape[1], : dt.shape[2]]
            zc, yc, xc = center
            # dt[(zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2 <= radius ** 2] = 0
            dt = np.min(
                (
                    dt,
                    np.max(
                        (
                            np.sqrt(
                                (zz - zc) ** 2
                                + (yy - yc) ** 2
                                + (xx - xc) ** 2
                            )
                            - radius,
                            np.zeros_like(dt),
                        ),
                        axis=0,
                    ),
                ),
                axis=0,
            )

        return radii, centers

    def run(self):
        # Perform morphometry extraction
        labels = measure.label(
            (
                self.data
                if self._label_choice == "All"
                else self.data == self._label_choice
            ),
            connectivity=3,
        )
        self._set_progress(20)
        labels = morphology.remove_small_objects(
            labels, min_size=10, connectivity=3
        )
        self._set_progress(40)
        props = measure.regionprops_table(
            labels,
            self._intensity_data,
            properties=[
                "label",
                "area",
                "inertia_tensor",
                "centroid",
                "local_centroid",
                "intensity_min",
                "intensity_max",
                "intensity_mean",
                "intensity_std",
                "bbox",
            ],
        )
        self._set_progress(50)

        num_labels = len(props["label"])
        props["max diameter"] = [0] * num_labels  # feret diameter
        props["sphericity"] = [0] * num_labels
        props["volume"] = props.pop("area")
        props["surface area"] = [0] * num_labels
        props["solidity"] = [0] * num_labels
        props["num. of inspheres"] = [0] * num_labels
        props["sum of insphere diameters"] = [0] * num_labels
        props["multiple sphericity"] = [0] * num_labels
        props["insphere fill ratio"] = [0] * num_labels

        additional_data = {}

        additional_data["intensity_label"] = measure.regionprops_table(
            labels, self.data, properties=["intensity_min"]
        )["intensity_min"].astype(np.uint32)
        additional_data["meshes"] = [None] * num_labels
        additional_data["positions"] = [None] * num_labels
        additional_data["upper_bounds"] = [None] * num_labels
        additional_data["inspheres"] = [()] * num_labels
        additional_data["centroids"] = np.column_stack(
            [props.pop(f"centroid-{i}") for i in range(3)]
        ).tolist()
        additional_data["local_centroids"] = np.column_stack(
            [props.pop(f"local_centroid-{i}") for i in range(3)]
        ).tolist()

        inertia_tensors = np.column_stack(
            [
                props.pop(f"inertia_tensor-{i}-{j}")
                for i in range(3)
                for j in range(3)
            ]
        ).reshape(-1, 3, 3)
        eigvals, eigvecs = np.linalg.eigh(inertia_tensors)
        props["orientation"] = eigvecs[:, :, 0].tolist()  # principal axis
        additional_data["major_axis_eigenvals"] = eigvals[:, 0]

        self._set_progress(65)

        for i, (loz, loy, lox, hiz, hiy, hix) in enumerate(
            zip(
                props.pop("bbox-0"),
                props.pop("bbox-1"),
                props.pop("bbox-2"),
                props.pop("bbox-3"),
                props.pop("bbox-4"),
                props.pop("bbox-5"),
                strict=False,
            )
        ):
            loz, loy, lox = max(0, loz - 1), max(0, loy - 1), max(0, lox - 1)
            hiz, hiy, hix = (
                min(labels.shape[0], hiz + 1),
                min(labels.shape[1], hiy + 1),
                min(labels.shape[2], hix + 1),
            )
            lbl = labels[loz:hiz, loy:hiy, lox:hix] == props["label"][i]
            additional_data["positions"][i] = np.array((loz, loy, lox))
            additional_data["upper_bounds"][i] = np.array((hiz, hiy, hix))

            # fit included sphere
            radii, centers = self.fit_spheres(lbl)
            props["num. of inspheres"][i] = len(radii)
            props["sum of insphere diameters"][i] = np.sum(radii) * 2
            props["insphere fill ratio"][i] = (
                4
                / 3
                * np.pi
                * np.sum(np.array(radii) ** 3)
                / props["volume"][i]
            )
            additional_data["inspheres"][i] = (
                radii,
                [c + np.array([loz, loy, lox]) for c in centers],
            )

            verts, faces, _, _ = measure.marching_cubes(lbl, level=0)
            props["surface area"][i] = measure.mesh_surface_area(verts, faces)
            additional_data["meshes"][i] = (verts, faces)
            props["sphericity"][i] = (
                36 * np.pi * props["volume"][i] ** 2
            ) ** (1 / 3) / props["surface area"][i]
            props["multiple sphericity"][i] = (
                4 * np.pi * sum(r**2 for r in radii) / props["surface area"][i]
            )

            try:
                hull = ConvexHull(verts)
                verts = verts[hull.vertices]

                # gridcoords = np.reshape(np.mgrid[tuple(map(slice, lbl.shape))], (3, -1))
                # coords_in_hull = morphology.convex_hull._check_coords_in_hull(gridcoords, hull.equations, tolerance=1e-10)
                # hull_volume = np.sum(coords_in_hull.reshape(lbl.shape))
                # props['solidity'][i] = props['volume'][i] / hull.volume
            except QhullError as e:
                print(
                    f"ConvexHull computation failed for label {props['label'][i]}: {e}"
                )
            props["max diameter"][i] = sqrt(pdist(verts, "sqeuclidean").max())

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"^Failed to get convex hull image\..*"
                )
                hull_volume = np.sum(morphology.convex_hull_image(lbl))
                if hull_volume > 0:
                    props["solidity"][i] = props["volume"][i] / hull_volume

            self._increment_progress(30 / num_labels)

        # Convert properties with metadata
        props = pd.DataFrame(props)
        additional_data = pd.DataFrame(additional_data)
        additional_data["feret_diameter_pixels"] = props["max diameter"].copy()
        if "pixelsize" in self._metadata:
            pixelsize = self._metadata["pixelsize"]
            additional_data["feret_diameter_raw"] = (
                props["max diameter"] * pixelsize
            )
            unit, pixelsize = extract_unit_and_value(pixelsize)

            props["max diameter"] *= pixelsize
            props["sum of insphere diameters"] *= pixelsize
            props["volume"] *= pixelsize**3
            props["surface area"] *= pixelsize**2
            props.rename(
                columns={
                    "max diameter": f"max diameter [{unit}]",
                    "sum of insphere diameters": f"sum of insphere diameters [{unit}]",
                    "volume": f"volume [{unit}³]",
                    "surface area": f"surface area [{unit}²]",
                },
                inplace=True,
            )

        if "low_cutoff" in self._metadata and "high_cutoff" in self._metadata:
            max_intensity = np.iinfo(self._intensity_data.dtype).max
            ri_factor = (
                self._metadata["high_cutoff"] - self._metadata["low_cutoff"]
            ) / max_intensity
            props["mean refractive index"] = (
                props.pop("intensity_mean") * ri_factor
                + self._metadata["low_cutoff"]
            )
            props["min refractive index"] = (
                props.pop("intensity_min") * ri_factor
                + self._metadata["low_cutoff"]
            )
            props["max refractive index"] = (
                props.pop("intensity_max") * ri_factor
                + self._metadata["low_cutoff"]
            )
            props["std refractive index"] = (
                props.pop("intensity_std") * ri_factor
            )
            if "factor_edensity" in self._metadata:
                props.insert(
                    3,
                    "mean ED [e⁻/Å³]",
                    props["mean refractive index"]
                    * self._metadata["factor_edensity"],
                )
                props["min ED [e⁻/Å³]"] = (
                    props.pop("min refractive index")
                    * self._metadata["factor_edensity"]
                )
                props["max ED [e⁻/Å³]"] = (
                    props.pop("max refractive index")
                    * self._metadata["factor_edensity"]
                )
                props["std ED [e⁻/Å³]"] = (
                    props.pop("std refractive index")
                    * self._metadata["factor_edensity"]
                )

        self._set_progress(100)

        self.finished.emit(
            {
                "props": props,
                "labels": labels,
                "label_choice": self._label_choice,
                "additional_data": additional_data,
            }
        )


class MorphometryWidget(PipelineWidget):

    def __init__(
        self,
        viewer: napari.Viewer,
        input_widget: Widget | None = None,
        intensity_input_widget: Widget | None = None,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            input_widget=input_widget,
            output_layer_suffix="_morphometry",
            output_layer_is_labels=True,
            input_layer_is_labels=True,
            **kwargs,
        )

        if input_widget is None:
            self._input_widget.label = "Label Data"
        elif not getattr(input_widget, "_output_layer_is_labels", False):
            raise NotImplementedError(
                "Widget preceding morphometry extraction must produce a label layer."
            )

        if intensity_input_widget is None:
            self._intensity_input_with_metadata = LayerPickerWithMetadata(
                viewer, label="Intensity Data"
            )
            self._intensity_input_widget = (
                self._intensity_input_with_metadata.layer_picker
            )
            self.append(self._intensity_input_with_metadata)
        else:
            self._intensity_input_widget = intensity_input_widget

        self._select_labels = RadioButtons(
            label="Choose label for extraction",
            choices=self._get_label_choice_options(),
            value="All",
        )
        self._select_labels.changed.connect(self._label_choice_changed)

        self._start_button = PushButton(text="Extract Morphometic Features")
        self._start_button.changed.connect(self._start_morphometry_extraction)

        self._progress_bar.label = "Morphometry Extraction Progress"

        self._tables = {}
        self._results = {}

        self.extend(
            [
                self._select_labels,
                self._start_button,
                self._progress_bar,
            ]
        )

    def _get_label_choice_options(self):
        label_layer = self._input_widget.value
        if label_layer is not None:
            return ["All"] + list(range(1, label_layer.data.max() + 1))
        return ["All"]

    def input_changed(self):
        if super().input_changed():
            self._select_labels.choices = self._get_label_choice_options()

    @debounce(10)
    def _label_choice_changed(self):
        if self._input_widget.value is None:
            return
        if self._select_labels.value == "All":
            self._input_widget.value.show_selected_label = False
        else:
            self._input_widget.value.show_selected_label = True
            self._input_widget.value.selected_label = self._select_labels.value

    def _start_morphometry_extraction(self):
        self._progress_bar.value = 0
        self._progress_bar.show()
        self.start_background_worker(
            MorphometryWorker,
            intensity_data=self._intensity_input_widget.value.data,
            label_choice=self._select_labels.value,
            metadata=self._intensity_input_widget.value.metadata,
        )

    def show_mesh(self, row, column):
        pass

    def process_background_result(self, result):
        self.output_data = result["labels"]
        self._input_widget.value.visible = False

        if result["label_choice"] in self._tables:
            cur_table_widget = self._tables.pop(result["label_choice"])
            if not sip.isdeleted(cur_table_widget):
                self._viewer.window.remove_dock_widget(cur_table_widget)
            del cur_table_widget
        table_name = "Morphometric Features of " + (
            "all labels"
            if result["label_choice"] == "All"
            else f'label {result["label_choice"]}'
        )
        table = ResultWidget(
            self._viewer,
            result,
            self._output_layer,
            self._intensity_input_widget.value,
            title=table_name,
        )
        table_dock_widget = self._viewer.window.add_dock_widget(
            table, name=table_name, area="right", tabify=True
        )
        # table_dock_widget.setFloating(True)
        self._tables[result["label_choice"]] = table_dock_widget

        self._progress_bar.hide()
