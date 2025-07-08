from collections.abc import Iterable
from functools import cached_property, wraps

import dask.array as da
import matplotlib.pyplot as plt
import meshio
import napari
import numpy as np
import pandas as pd
import sip
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QWidgetAction,
)

from ._utils._util_funcs import debounce, get_unit_factor


class ResultWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.

    Adapted from https://github.com/haesleinhuepf/napari-skimage-regionprops/blob/master/napari_skimage_regionprops/_table.py
    """

    def __init__(
        self,
        viewer: "napari.Viewer",
        results: dict,
        labels_layer: "napari.layers.Labels",
        intensity_layer: "napari.layers.Image",
        title: str = "",
    ):
        super().__init__()

        self._labels_layer = labels_layer
        self._intensity_layer = intensity_layer
        self._viewer = viewer

        self._label_choice = results["label_choice"]
        self._raw_table = results["props"]
        self._raw_additional_data = results["additional_data"]
        self._raw_labels = results["labels"]

        self._table = self._raw_table
        self._additional_data = self._raw_additional_data

        main_view_label = QLabel("Individual object properties<hr>")
        main_view_label.setAlignment(Qt.AlignCenter)
        main_view_label.setStyleSheet("font-size: 11pt; padding-top: 20px;")
        self._view = QTableWidget()
        self._view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        group_view_label = QLabel("Group properties<hr>")
        group_view_label.setAlignment(Qt.AlignCenter)
        group_view_label.setStyleSheet("font-size: 11pt; padding-top: 20px;")
        self._group_view = QTableWidget()
        self._group_view.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._group_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.update_content()

        self._view.clicked.connect(self._clicked_table)
        self._view.itemSelectionChanged.connect(self._selection_changed)
        labels_layer.events.selected_label.connect(self._clicked_labels)

        self._view.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self._view.customContextMenuRequested.connect(
            self._row_context_menu_requested
        )
        self._view.verticalHeader().customContextMenuRequested.connect(
            self._row_context_menu_requested
        )

        self._view.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self._view.horizontalHeader().customContextMenuRequested.connect(
            self._header_context_menu_requested
        )

        filter_label = QLabel("Filter out objects smaller than")
        self.filter_diameter_picker = QSpinBox()
        self.filter_diameter_picker.setRange(0, 1000000)
        self.filter_diameter_picker.setValue(0)
        self.filter_diameter_picker.editingFinished.connect(
            self.filter_diameter
        )
        self.unit_selection = QComboBox()
        self.unit_selection.setMaximumWidth(70)
        self.unit_selection.addItems(["pm", "nm", "µm", "mm", "m"])
        self.unit_selection.setCurrentText("nm")
        filter_button = QPushButton("Go")
        filter_button.setMaximumWidth(50)
        filter_button.clicked.connect(self.filter_diameter)

        roi_button = QPushButton("Show ROI intensity")
        roi_button.clicked.connect(self._show_roi)
        spheres_button = QPushButton("Show inspheres")
        spheres_button.clicked.connect(self._show_inspheres)

        surface_button = QPushButton("Show surface meshes")
        surface_button.clicked.connect(self._show_full_mesh)
        orientation_button = QPushButton("Show orientation vectors")
        orientation_button.clicked.connect(self._show_orientation_vectors)

        export_surface_button = QPushButton("Export surface meshes...")
        export_surface_button.clicked.connect(self._export_full_mesh)

        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_clicked)
        save_button = QPushButton("Save as csv...")
        save_button.clicked.connect(self._save_clicked)

        group_copy_button = QPushButton("Copy to clipboard")
        group_copy_button.clicked.connect(self._group_copy_clicked)
        group_save_button = QPushButton("Save as csv...")
        group_save_button.clicked.connect(self._group_save_clicked)

        self.setWindowTitle(title)
        self.setLayout(QGridLayout())

        filter_volume_widget = QWidget()
        filter_volume_widget.setLayout(QHBoxLayout())
        filter_volume_widget.layout().addWidget(filter_label)
        filter_volume_widget.layout().addWidget(self.filter_diameter_picker)
        filter_volume_widget.layout().addWidget(self.unit_selection)
        filter_volume_widget.layout().addWidget(filter_button)
        filter_volume_widget.layout().setContentsMargins(0, 0, 0, 0)
        filter_volume_widget.layout().setSpacing(10)

        layers_widget = QWidget()
        layers_widget.setLayout(QHBoxLayout())
        layers_widget.layout().addWidget(roi_button)
        layers_widget.layout().addWidget(spheres_button)
        layers_widget.layout().setSpacing(3)
        layers_widget.layout().setContentsMargins(0, 0, 0, 0)

        surface_widget = QWidget()
        surface_widget.setLayout(QHBoxLayout())
        surface_widget.layout().addWidget(surface_button)
        surface_widget.layout().addWidget(orientation_button)
        surface_widget.layout().setSpacing(3)
        surface_widget.layout().setContentsMargins(0, 0, 0, 0)

        main_action_widget = QWidget()
        main_action_widget.setLayout(QHBoxLayout())
        main_action_widget.layout().addWidget(copy_button)
        main_action_widget.layout().addWidget(save_button)
        main_action_widget.layout().setSpacing(3)
        main_action_widget.layout().setContentsMargins(0, 0, 0, 0)
        group_action_widget = QWidget()
        group_action_widget.setLayout(QHBoxLayout())
        group_action_widget.layout().addWidget(group_copy_button)
        group_action_widget.layout().addWidget(group_save_button)
        group_action_widget.layout().setSpacing(3)
        group_action_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().addWidget(filter_volume_widget)
        self.layout().addWidget(layers_widget)
        self.layout().addWidget(surface_widget)
        self.layout().addWidget(export_surface_button)
        self.layout().addWidget(group_view_label)
        self.layout().addWidget(group_action_widget)
        self.layout().addWidget(self._group_view)
        self.layout().addWidget(main_view_label)
        self.layout().addWidget(main_action_widget)
        self.layout().addWidget(self._view)

    def add_context_menu_action(
        self, action_name: str, callback: callable, menu: QMenu
    ):
        action = QAction(action_name, menu)
        action.triggered.connect(callback)
        menu.addAction(action)

    def _row_context_menu_requested(self, pos: QPoint):
        row = self._view.rowAt(pos.y())
        if row < 0:
            return
        label = self._table["label"][row]
        menu = QMenu(self._view)

        title_label = QLabel(f"Object {label}")
        title_label.setStyleSheet(
            "font-size: 9pt; font-weight: bold; padding: 0px 15px;"
        )
        title_action = QWidgetAction(menu)
        title_action.setDefaultWidget(title_label)

        show_submenu = QMenu("Show...", menu)
        export_submenu = QMenu("Export...", menu)

        self.add_context_menu_action(
            "Show Label", lambda: self._show_one_label(row), show_submenu
        )
        self.add_context_menu_action(
            "Show Intensity",
            lambda: self._show_one_intensity(row),
            show_submenu,
        )
        self.add_context_menu_action(
            "Show Inspheres",
            lambda: self._show_one_inspheres(row),
            show_submenu,
        )
        self.add_context_menu_action(
            "Show Mesh", lambda: self._show_one_mesh(row), show_submenu
        )
        self.add_context_menu_action(
            "Show All", lambda: self._show_one_all(row), show_submenu
        )
        self.add_context_menu_action(
            "Export Mesh...",
            lambda: self._export_one_mesh(row),
            export_submenu,
        )

        menu.addAction(title_action)
        menu.addSeparator()
        menu.addMenu(show_submenu)
        menu.addMenu(export_submenu)

        menu.exec_(self._view.mapToGlobal(pos))

    def _header_context_menu_requested(self, pos: QPoint):
        header = self._view.horizontalHeader()
        col_idx = header.logicalIndexAt(pos)
        if col_idx < 0:
            return
        column_name = self._table.columns[col_idx]
        menu = QMenu(self._view)

        title_label = QLabel(f"{column_name}")
        title_label.setStyleSheet(
            "font-size: 9pt; font-weight: bold; padding: 0px 15px;"
        )
        title_action = QWidgetAction(menu)
        title_action.setDefaultWidget(title_label)

        scatter_submenu = QMenu("Scatter against", menu)
        for other_col in self._table.columns:
            if other_col != column_name:
                scatter_action = QAction(other_col, scatter_submenu)
                scatter_action.triggered.connect(
                    lambda checked=False, oc=other_col: self._scatter_columns(
                        self._view.currentRow(), oc, column_name
                    )
                )
                scatter_submenu.addAction(scatter_action)

        boxplot_action = QAction("Boxplot", menu)
        boxplot_action.triggered.connect(
            lambda: self._boxplot_column(self._view.currentRow(), column_name)
        )
        histogram_action = QAction("Histogram", menu)
        histogram_action.triggered.connect(
            lambda: self._histogram_column(
                self._view.currentRow(), column_name
            )
        )

        menu.addAction(title_action)
        menu.addSeparator()
        menu.addMenu(scatter_submenu)
        menu.addAction(boxplot_action)
        menu.addAction(histogram_action)
        menu.exec_(header.mapToGlobal(pos))

    @debounce(50)
    def _selection_changed(self):
        r = self._view.selectedRanges()
        if self._labels_layer.show_selected_label and (
            len(r) == 0
            or len(r) > 1
            or r[0].rowCount() > 1
            or r[0].rowCount() == 0
        ):
            self._labels_layer.show_selected_label = False
            if hasattr(self, "_inspheres_layer"):
                self._inspheres_layer.show_selected_label = False

    @debounce(50)
    def _clicked_table(self, index):
        if "label" in self._table:
            row = index.row()
            label = self._table["label"][row]
            print("Table clicked, set label", label)
            self._labels_layer.selected_label = label
            if not self._labels_layer.show_selected_label:
                self._labels_layer.show_selected_label = True
            if hasattr(self, "_inspheres_layer"):
                self._inspheres_layer.selected_label = label
                if not self._inspheres_layer.show_selected_label:
                    self._inspheres_layer.show_selected_label = True

    @debounce(50)
    def _clicked_labels(self):
        if "label" in self._table and hasattr(
            self._labels_layer, "selected_label"
        ):
            lbl = self._labels_layer.selected_label
            try:
                lbl_idx = self._table.index[self._table["label"] == lbl][0]
                if lbl_idx != self._view.currentRow():
                    self._view.selectRow(lbl_idx)
            except IndexError:
                print(f"Label {lbl} not found in table")

    def filter_diameter(self):
        val = self.filter_diameter_picker.value()
        unit = self.unit_selection.currentText()
        val *= get_unit_factor(unit)
        index_filter = np.array(
            [k >= val for k in self._raw_additional_data["feret_diameter_raw"]]
        )
        self._table = self._raw_table[index_filter]
        self._table.index = pd.RangeIndex(
            start=0, stop=self._table.shape[0], step=1
        )
        self._additional_data = self._raw_additional_data[index_filter]
        self._additional_data.index = pd.RangeIndex(
            start=0, stop=self._additional_data.shape[0], step=1
        )
        self.update_content()

        removed_labels = self._raw_table["label"][~index_filter].to_numpy(
            dtype=self._raw_labels.dtype
        )
        self._labels_layer.data = np.where(
            np.isin(self._raw_labels, removed_labels), 0, self._raw_labels
        )
        if hasattr(self, "_inspheres_layer"):
            self._show_inspheres()
        if hasattr(self, "_surface_layer"):
            self._show_full_mesh()

        self.__dict__.pop("full_mesh", None)

    @cached_property
    def full_mesh(self):
        verts, faces = zip(*self._additional_data["meshes"], strict=False)
        full_faces = [
            f + vert_shift
            for f, vert_shift in zip(
                faces,
                np.cumsum([0] + [k.shape[0] for k in verts[:-1]]),
                strict=False,
            )
        ]
        return (
            np.concatenate(
                [
                    v + p
                    for v, p in zip(
                        verts, self._additional_data["positions"], strict=False
                    )
                ]
            ),
            np.concatenate(full_faces),
        )

    def _show_roi(self):
        masked_intensity_layer_name = (
            f"{self._intensity_layer.name}_{self._label_choice}_ROI"
        )
        if isinstance(self._intensity_layer.data, da.Array):
            masked_intensity_data = self._intensity_layer.data.compute()
        else:
            masked_intensity_data = self._intensity_layer.data.copy()
        masked_intensity_data[self._labels_layer.data == 0] = 0
        if masked_intensity_layer_name not in self._viewer.layers:
            self._viewer.add_image(
                masked_intensity_data, name=masked_intensity_layer_name
            )
        else:
            self._viewer.layers[masked_intensity_layer_name].data = (
                masked_intensity_data
            )

    def _show_inspheres(self):
        inspheres_layer_name = (
            f"{self._intensity_layer.name}_{self._label_choice}_inspheres"
        )
        inspheres_data = np.zeros_like(self._labels_layer.data)
        z_size, y_size, x_size = inspheres_data.shape
        for label, inspheres in zip(
            self._table["label"],
            self._additional_data["inspheres"],
            strict=False,
        ):
            for radius, center in zip(*inspheres, strict=False):
                zc, yc, xc = center
                zmin, zmax = max(0, int(zc - radius)), min(
                    z_size, int(zc + radius + 1)
                )
                ymin, ymax = max(0, int(yc - radius)), min(
                    y_size, int(yc + radius + 1)
                )
                xmin, xmax = max(0, int(xc - radius)), min(
                    x_size, int(xc + radius + 1)
                )
                # zmin, ymin, xmin = np.floor(center - radius).astype(int)
                # zmax, ymax, xmax = np.ceil(center + radius).astype(int)
                z, y, x = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
                inspheres_data[zmin:zmax, ymin:ymax, xmin:xmax][
                    (z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2 <= radius**2
                ] = label

        if inspheres_layer_name not in self._viewer.layers:
            self._viewer.add_labels(inspheres_data, name=inspheres_layer_name)
        else:
            self._viewer.layers[inspheres_layer_name].data = inspheres_data
        self._inspheres_layer = self._viewer.layers[inspheres_layer_name]

    def _show_orientation_vectors(self):
        orientation_layer_name = (
            f"{self._intensity_layer.name}_{self._label_choice}_orientation"
        )
        orientation_vectors = (
            np.array(self._table["orientation"].tolist())
            * self._additional_data["feret_diameter_pixels"].to_numpy()[
                :, None
            ]
        )  # np.sqrt(self._additional_data['major_axis_eigenvals'].to_numpy())[:, None]
        positions = np.array(self._additional_data["centroids"].tolist())
        orientation_data = np.stack([positions, orientation_vectors], axis=1)

        if orientation_layer_name in self._viewer.layers:
            self._viewer.layers[orientation_layer_name].data = orientation_data
            self._viewer.layers[orientation_layer_name].visible = True
        else:
            self._viewer.add_vectors(
                orientation_data, name=orientation_layer_name, edge_width=2.0
            )
        self._orientation_layer = self._viewer.layers[orientation_layer_name]

    def _show_full_mesh(self):
        surface_layer_name = (
            f"{self._intensity_layer.name}_{self._label_choice}_meshes"
        )
        if surface_layer_name in self._viewer.layers:
            self._viewer.layers[surface_layer_name].data = self.full_mesh
            self._viewer.layers[surface_layer_name].visible = True
        else:
            self._viewer.add_surface(self.full_mesh, name=surface_layer_name)
        self._surface_layer = self._viewer.layers[surface_layer_name]

    def _export_full_mesh(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export surface mesh...",
                ".",
                "3D file (*.stl, *.obj, *.ply, *.vtk)",
            )
        if filename:
            meshio.write_points_cells(
                filename, self.full_mesh[0], [("triangle", self.full_mesh[1])]
            )

    @staticmethod
    def with_extra_viewer(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "_extra_viewer") or sip.isdeleted(
                self._extra_viewer.window._qt_window
            ):  # TODO access to _qt_window will be removed in future versions of napari
                self._extra_viewer = napari.Viewer(
                    ndisplay=3, title="Individual object viewer"
                )
                self._extra_viewer.window.resize(1000, 800)
            layers = func(self, *args, **kwargs)
            if not isinstance(layers, list):
                layers = [layers]
            for layer in self._extra_viewer.layers:
                layer.visible = layer in layers
            self._extra_viewer.layers.selection = layers
            self._extra_viewer.window._qt_window.raise_()

        return wrapper

    @with_extra_viewer.__func__
    def _show_one_mesh(self, row):
        mesh_name = f'Mesh of object {self._table["label"][row]}'
        if mesh_name not in self._extra_viewer.layers:
            self._extra_viewer.add_surface(
                self._additional_data["meshes"][row], name=mesh_name
            )
        return self._extra_viewer.layers[mesh_name]

    @with_extra_viewer.__func__
    def _show_one_intensity(self, row):
        label = self._table["label"][row]
        intensity_name = f"Intensity around object {label}"
        if intensity_name not in self._extra_viewer.layers:
            loz, loy, lox = self._additional_data["positions"][row]
            hiz, hiy, hix = self._additional_data["upper_bounds"][row]
            intensity_data = self._intensity_layer.data[
                loz:hiz, loy:hiy, lox:hix
            ]
            if isinstance(intensity_data, da.Array):
                intensity_data = intensity_data.compute()
            foreground_mean = np.mean(
                intensity_data[
                    self._labels_layer.data[loz:hiz, loy:hiy, lox:hix] == label
                ]
            )
            background_mean = np.mean(
                intensity_data[
                    self._labels_layer.data[loz:hiz, loy:hiy, lox:hix] == 0
                ]
            )
            rendering_mode = (
                "mip" if foreground_mean > background_mean else "minip"
            )
            self._extra_viewer.add_image(intensity_data, name=intensity_name)
            self._extra_viewer.layers[intensity_name].rendering = (
                rendering_mode
            )
        return self._extra_viewer.layers[intensity_name]

    @with_extra_viewer.__func__
    def _show_one_inspheres(self, row):
        label = self._table["label"][row]
        inspheres_name = f"Inspheres of object {label}"
        if inspheres_name not in self._extra_viewer.layers:
            z_size, y_size, x_size = (
                self._additional_data["upper_bounds"][row]
                - self._additional_data["positions"][row]
            )
            inspheres_data = np.zeros(
                (z_size, y_size, x_size), dtype=np.uint32
            )
            for radius, center in zip(
                *self._additional_data["inspheres"][row], strict=False
            ):
                zc, yc, xc = center - self._additional_data["positions"][row]
                zmin, zmax = max(0, int(zc - radius)), min(
                    z_size, int(zc + radius + 1)
                )
                ymin, ymax = max(0, int(yc - radius)), min(
                    y_size, int(yc + radius + 1)
                )
                xmin, xmax = max(0, int(xc - radius)), min(
                    x_size, int(xc + radius + 1)
                )
                z, y, x = np.ogrid[zmin:zmax, ymin:ymax, xmin:xmax]
                inspheres_data[zmin:zmax, ymin:ymax, xmin:xmax][
                    (z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2 <= radius**2
                ] = label
            self._extra_viewer.add_labels(inspheres_data, name=inspheres_name)
        return self._extra_viewer.layers[inspheres_name]

    @with_extra_viewer.__func__
    def _show_one_label(self, row):
        label = self._table["label"][row]
        label_name = f"Object {label}"
        if label_name not in self._extra_viewer.layers:
            loz, loy, lox = self._additional_data["positions"][row]
            hiz, hiy, hix = self._additional_data["upper_bounds"][row]
            label_data = self._labels_layer.data[loz:hiz, loy:hiy, lox:hix]
            label_data[label_data != label] = 0
            self._extra_viewer.add_labels(label_data, name=label_name)
        return self._extra_viewer.layers[label_name]

    @with_extra_viewer.__func__
    def _show_one_all(self, row):
        layers = [
            self._show_one_intensity.__wrapped__(self, row),
            self._show_one_inspheres.__wrapped__(self, row),
            self._show_one_label.__wrapped__(self, row),
        ]
        mesh_layer = self._show_one_mesh.__wrapped__(self, row)
        mesh_layer.opacity = 0.5
        layers.append(mesh_layer)
        return layers

    def _export_one_mesh(self, row):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export surface mesh...",
            ".",
            "3D file (*.stl, *.obj, *.ply, *.vtk)",
        )
        if filename:
            meshio.write_points_cells(
                filename,
                self._additional_data["meshes"][row][0],
                [("triangle", self._additional_data["meshes"][row][1])],
            )

    def _export_one_intensity(self, row):
        label = self._table["label"][row]
        loz, loy, lox = self._additional_data["positions"][row]
        hiz, hiy, hix = self._additional_data["upper_bounds"][row]
        intensity_data = self._intensity_layer.data[loz:hiz, loy:hiy, lox:hix]
        if isinstance(intensity_data, da.Array):
            intensity_data = intensity_data.compute()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export intensity around object {label}...",
            ".",
            "Numpy array (*.npy)",
        )
        if filename:
            np.save(filename, intensity_data)

    def _save_clicked(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save as csv...", ".", "*.csv"
            )
        if filename:
            self._table.to_csv(filename)

    def _copy_clicked(self):
        self._table.to_clipboard()

    def _group_save_clicked(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save as csv...", ".", "*.csv"
            )
        if filename:
            self._group_table.to_csv(filename)

    def _group_copy_clicked(self):
        self._group_table.to_clipboard()

    def calculate_group_props(self):
        diam_col_name = self._table.columns[
            self._table.columns.str.startswith("max diameter")
        ][0]

        helper_table = {
            "intensity_label": self._additional_data["intensity_label"],
            diam_col_name: self._table[diam_col_name],
        }

        if "mean ED [e⁻/Å³]" in self._table.columns:
            helper_table["ED [e⁻/Å³]"] = self._table["mean ED [e⁻/Å³]"]
        else:
            helper_table["intensity"] = self._table["intensity_mean"]

        vol_col_names = self._table.columns[
            self._table.columns.str.startswith("volume")
        ]
        if len(vol_col_names) > 0:
            vol_col_name = vol_col_names[0]
            helper_table[vol_col_name] = self._table[vol_col_name]

        # causes issues with std()
        # if 'orientation' in self._table.columns:
        #     helper_table['orientation'] = [np.array(x) for x in self._table['orientation'].tolist()]

        helper_table = pd.DataFrame(helper_table)
        self._group_table = helper_table.groupby("intensity_label").agg(
            ["mean", "std"]
        )
        self._group_table.columns = [
            " ".join(col[::-1]) if isinstance(col, tuple) else col
            for col in self._group_table.columns
        ]

    def set_table_content(self, table, view, resize_height=False):
        view.clear()
        view.setRowCount(table.shape[0])
        view.setColumnCount(table.shape[1])

        for i, column in enumerate(table.columns):
            view.setHorizontalHeaderItem(i, QTableWidgetItem(column))
            for j, value in enumerate(table[column]):
                if isinstance(value, float):
                    value_repr = f"{value:8g}"
                elif isinstance(value, Iterable):
                    value_repr = f"[{', '.join(f'{v:2g}' for v in value)}]"
                else:
                    value_repr = str(value)
                view.setItem(j, i, QTableWidgetItem(value_repr))

        view.resizeColumnsToContents()

        if resize_height:
            height = view.horizontalHeader().height() + sum(
                view.rowHeight(i) for i in range(view.rowCount())
            )
            view.setFixedHeight(height + 20)

    def update_content(self):
        self.set_table_content(self._table, self._view)
        self.calculate_group_props()
        self.set_table_content(self._group_table, self._group_view, True)

    def _scatter_columns(self, row, col1, col2):
        print(f"Scatter columns {col1} and {col2} clicked. Row = {row}")

        fig, ax = plt.subplots()
        ax.scatter(self._table[col1], self._table[col2])
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Scatter plot of {col2} vs {col1}")
        fig.show()

    def _boxplot_column(self, row, col):
        print(f"Boxplot column {col} clicked. Row = {row}")

        fig, ax = plt.subplots()
        helper_table = pd.DataFrame(
            {
                "intensity_label": self._additional_data["intensity_label"],
                col: self._table[col],
            }
        )
        helper_table.boxplot(column=col, by="intensity_label", ax=ax)
        ax.set_title(f"Boxplot of {col} grouped by intensity")
        ax.set_xlabel("Intensity Label")
        ax.set_ylabel(col)
        fig.suptitle("")
        fig.show()

    def _histogram_column(self, row, col):
        print(f"Histogram column {col} clicked. Row = {row}")

        fig, ax = plt.subplots()
        ax.hist(self._table[col], bins=100)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {col}")
        fig.show()
