# import os
# os.environ["NAPARI_ASYNC"] = "1"

from napari import Viewer, run

viewer = Viewer(ndisplay=3)
viewer.open_sample("microfossil-biogenicity", "hg1_delta")
# viewer.open("data/KB1_delta_cropped_filtered.tif", name="KB1_cropped_filtered")
# viewer.open("data/HG1_delta_cropped_filtered_cropped_segmented.tif")
# viewer.open("data/KB1_cropped.tif", name="KB1_cropped")
# viewer.window.add_plugin_dock_widget(
#     "microfossil-biogenicity",
#     "2. Filter",
# )
# viewer.window.add_plugin_dock_widget(
#     "microfossil-biogenicity",
#     "4. Morphometry",
# )
viewer.window.add_plugin_dock_widget(
    "microfossil-biogenicity",
    "Full Pipeline",
)

run()
