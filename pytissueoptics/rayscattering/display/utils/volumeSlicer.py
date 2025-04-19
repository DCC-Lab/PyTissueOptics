# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2009, Enthought, Inc.
# License: BSD Style.

import numpy as np

from traits.api import HasTraits, Instance, Array, \
    on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel

try:
    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    VIEW = View(HGroup(
        Group(
            Item('scene_y',
                 editor=SceneEditor(scene_class=Scene),
                 height=250, width=300),
            Item('scene_z',
                 editor=SceneEditor(scene_class=Scene),
                 height=250, width=300),
            show_labels=False,
        ),
        Group(
            Item('scene_x',
                 editor=SceneEditor(scene_class=Scene),
                 height=250, width=300),
            Item('scene3d',
                 editor=SceneEditor(scene_class=MayaviScene),
                 height=250, width=300),
            show_labels=False,
        ),
    ),
        resizable=True,
        title='Volume Slicer',
    )
except Exception as e:
    VIEW = e


class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array()

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    _axis_names = dict(x=0, y=1, z=2)

    view = VIEW

    #---------------------------------------------------------------------------
    def __init__(self, hist3D: np.ndarray, colormap: str = 'viridis', interpolate=False, **traits):
        self._colormap = colormap
        self._cameraView = {"azimuth": -30, "zenith": 215, "distance": None, "pointingTowards": None, "roll": -0}
        self._cameraPitch = -3
        self._interpolate = interpolate

        if isinstance(VIEW, Exception):
            raise VIEW

        super(VolumeSlicer, self).__init__(data=hist3D, **traits)

        # Force the creation of the image_plane_widgets:
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            if not self._interpolate:
                ipw.ipw.texture_interpolate = "off"
                ipw.ipw.set_input_data(ipw.ipw._get_input())

    def show(self):
        self.configure_traits()

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _data_src3d_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src3d,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name, colormap=self._colormap)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbaks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(self.data_src3d,
                        figure=self.scene3d.mayavi_scene,
                                        colormap=self._colormap)
        # self.scene3d.mlab.view(40, 50)
        self.scene3d.mlab.view(*self._cameraView.values())
        # self.scene3d.mlab.pitch(self._cameraPitch)

        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            # Turn the interaction off
            ipw.ipw.interaction = 0

        self.scene3d.scene.background = (0.11, 0.11, 0.11)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
                            self.data_src3d.mlab_source.dataset,
                            figure=scene.mayavi_scene)
        ipw = mlab.pipeline.image_plane_widget(
                            outline,
                            plane_orientation='%s_axes' % axis_name, colormap=self._colormap)
        if not self._interpolate:
            ipw.ipw.texture_interpolate = "off"
            ipw.ipw.set_input_data(ipw.ipw._get_input())
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0
        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            position = obj.GetCurrentCursorPosition()
            for other_axis, axis_number in self._axis_names.items():
                if other_axis == axis_name:
                    continue
                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
                ipw3d.ipw.slice_position = position[axis_number]

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                    self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=( 0, 90),
                     y=(90, 90),
                     z=( 0,  0),
                     )
        # scene.mlab.view(*views[axis_name])

        scene.mlab.view(*self._cameraView.values())
        scene.mlab.pitch(self._cameraPitch)

        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0.11, 0.11, 0.11)


    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


if __name__ == '__main__':
    # Volume Slicer example with some data
    x, y, z = np.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
    data = np.sin(3 * x) / x + 0.05 * z ** 2 + np.cos(3 * y)

    m = VolumeSlicer(data)
    m.show()
