from napari import Viewer, run

viewer = Viewer(ndisplay=3)
viewer.open_sample("microfossil-biogenicity", "hg2_delta")
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "microfossil-biogenicity",
    "Extract mophometry features for biogenicity evaluation",
)
# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()
run()
