"""
This is the main class where NEURON cells are imported into Blender, where they are positioned and oriented within
bulbar layers. Transformed cell coordinates are saved for later instatiation by NEURON during simulations.
Synapse locations are identified and saved into synapse set files.
"""

import bpy, sys, os, time, re
import mathutils
import random
import os, sys
import bpy
import numpy as np
from collections import OrderedDict
from mathutils import Euler, Vector
import random
from math import floor, pi, acos
from fnmatch import fnmatch
from heapq import heappop, heappush
import json
sys.path.append(os.getcwd())
from olfactorybulb import slices
from olfactorybulb.epli import (
    DEFAULT_EPLI_MODEL_KEY,
    EPLI_GROUP_NAME,
    default_slice_group_colors,
    default_slice_group_names,
    default_slice_synapse_blueprints,
    epli_population_enabled,
    resolve_epli_model_spec,
)
from olfactorybulb.slicebuilder.config import slice_builder_env_kwargs
from blenderneuron.blender.utils import fast_get, make_safe_filename
from blenderneuron.blender.views.vectorconfinerview import VectorConfinerView
from blenderneuron.blender.views.synapseformerview import SynapseFormerView, SynapsePair, SynapseTerminal
import blenderneuron
import blenderneuron.blender

'''
Sources:
Kikuta et. al. 2013 - TC soma distance from Glom center ~200 um
Witman and Greer 2007 - GC spine reach - from digitized figure 5.5 um
'''

def _auto_start_handler_list():
    """
    Return the best available Blender app-handler list for deferred startup.

    Older code used ``scene_update_post`` directly, but newer/other builds only
    expose ``depsgraph_update_post``. We keep the deferred-start behavior so the
    BlenderNEURON addon has time to finish registering types before the slice
    builder touches them.
    """
    handlers = bpy.app.handlers
    for name in ("scene_update_post", "depsgraph_update_post"):
        handler_list = getattr(handlers, name, None)
        if handler_list is not None:
            return handler_list
    raise AttributeError("No compatible Blender app handler list found for slice-builder auto-start.")


def _matmul(left, right):
    """Compatibility wrapper for Blender matrix/vector multiplication."""
    try:
        return left @ right
    except TypeError:
        return left * right


def _evaluated_mesh_obj(mesh_obj):
    """Return the depsgraph-evaluated form of a mesh-like object when available."""
    if bpy.app.background:
        return mesh_obj
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return mesh_obj.evaluated_get(depsgraph)
    except Exception:
        return mesh_obj


def _object_vertex_points_global(mesh_obj):
    """Approximate one object surface by its global-space vertex cloud."""
    matrix_world = mesh_obj.matrix_world
    return np.array([
        np.array(_matmul(matrix_world, vertex.co))
        for vertex in getattr(mesh_obj.data, "vertices", [])
    ])


def _select_object(obj, select=True):
    """Compatibility wrapper for selecting Blender objects across API versions."""
    if hasattr(obj, "select_set"):
        obj.select_set(select)
    else:
        obj.select = select


def _selected_objects():
    """Return the current Blender selection in a version-tolerant way."""
    try:
        return list(bpy.context.selected_objects)
    except Exception:
        scene_objects = getattr(bpy.context.scene, "objects", [])
        return [obj for obj in scene_objects if getattr(obj, "select", False)]


def _set_active_object(obj):
    """Compatibility wrapper for making one object active."""
    try:
        bpy.context.view_layer.objects.active = obj
        return
    except Exception:
        pass

    try:
        bpy.context.scene.objects.active = obj
    except Exception:
        pass


def _scene_link_object(obj):
    """Link an object into the active scene using the supported collection API."""
    try:
        bpy.context.scene.collection.objects.link(obj)
        return
    except Exception:
        pass

    try:
        bpy.context.scene.objects.link(obj)
    except Exception:
        pass


def _scene_unlink_object(obj):
    """Unlink an object from the scene or its owning collections."""
    collections = list(getattr(obj, "users_collection", []))
    if collections:
        for collection in collections:
            try:
                collection.objects.unlink(obj)
            except Exception:
                pass
        return

    try:
        bpy.context.scene.objects.unlink(obj)
    except Exception:
        pass


def _scene_update():
    """Flush scene/view-layer transforms for whichever Blender API is present."""
    try:
        bpy.context.view_layer.update()
        return
    except Exception:
        pass

    try:
        bpy.context.scene.update()
    except Exception:
        pass


def _view3d_context_override(selected_object=None):
    """
    Build a 3D-view operator override for interactive operations when available.

    Background exports often have no usable 3D window context; callers should be
    prepared to receive ``None`` and skip view-only operators.
    """
    window_manager = getattr(bpy.context, "window_manager", None)
    windows = list(getattr(window_manager, "windows", [])) if window_manager is not None else []
    if not windows:
        return None

    window = windows[0]
    screen = getattr(window, "screen", None)
    if screen is None:
        screen = bpy.data.screens[0] if len(bpy.data.screens) > 0 else None
    if screen is None:
        return None

    area = None
    region = None
    for candidate_area in screen.areas:
        if candidate_area.type != 'VIEW_3D':
            continue
        area = candidate_area
        for candidate_region in candidate_area.regions:
            if candidate_region.type == 'WINDOW':
                region = candidate_region
                break
        if region is not None:
            break

    if area is None or region is None:
        return None

    override = {
        "window": window,
        "screen": screen,
        "area": area,
        "region": region,
        "scene": bpy.context.scene,
        "edit_object": None,
        "gpencil_data": None,
    }

    if selected_object is not None:
        override["object"] = selected_object
        override["active_object"] = selected_object
        override["edit_object"] = selected_object

    return override


def _align_object_towards(ob, pt_global, max_angle):
    """Rotate one curve object incrementally toward a target point."""
    max_angle = max_angle / 180.0 * 3.141592

    ob_mw = ob.matrix_world
    end = ob.data.splines[0].bezier_points[-1].co
    start = ob.data.splines[0].bezier_points[0].co
    desired = _matmul(ob_mw.inverted(), pt_global)
    v_start = end - start
    v_des = desired - start
    q = v_start.rotation_difference(v_des).to_euler()
    q = Euler(list(map(lambda angle: min(max(angle, -max_angle), max_angle), q))).to_quaternion()

    ob.matrix_basis = _matmul(ob.matrix_basis.copy(), q.to_matrix().to_4x4())

    if ob.parent is None:
        ob.matrix_world = ob.matrix_basis
    else:
        ob.matrix_world = _matmul(_matmul(ob.parent.matrix_world, ob.matrix_parent_inverse), ob.matrix_basis)

    _scene_update()


def _confine_between_meshes(curve_obj, start_mesh, end_mesh, height_low, height_high, max_angle, iters=11):
    """Compatibility implementation of BlenderNEURON's layer-confiner rotation step."""
    height_fraction = height_low + (height_high - height_low) * random.random()
    sec_start_loc = _matmul(curve_obj.matrix_world, curve_obj.data.splines[0].bezier_points[0].co)

    for _ in range(iters):
        tip_loc = _matmul(curve_obj.matrix_world, curve_obj.data.splines[0].bezier_points[-1].co)
        closest_on_start, _dist_to_start = SliceBuilderBlender.closest_point_on_object(np.array(tip_loc), start_mesh)
        closest_on_end, _dist_to_end = SliceBuilderBlender.closest_point_on_object(np.array(tip_loc), end_mesh)

        closest_on_start = Vector(closest_on_start)
        closest_on_end = Vector(closest_on_end)

        vec_start2end = (closest_on_end - closest_on_start).normalized()
        vec_start2tip = (tip_loc - closest_on_start).normalized()
        angle = acos(min(max(vec_start2end.dot(vec_start2tip), -1), 1)) * 180 / pi

        above = angle < 90 - 0.02
        if not above:
            vec_start2tip *= -1

        height = (closest_on_end - closest_on_start).length
        align_target = closest_on_start + vec_start2end * height * height_fraction
        _align_object_towards(curve_obj, align_target, max_angle / iters)


def _confine_curve(curve_obj, mesh, outer_mesh, name_pattern, height_range, max_angle):
    """Recursively confine a curve-object tree between two layer meshes."""
    if name_pattern is None or fnmatch(curve_obj.name, name_pattern):
        _confine_between_meshes(curve_obj, mesh, outer_mesh, height_range[0], height_range[1], max_angle)

    for child in curve_obj.children:
        _confine_curve(child, mesh, outer_mesh, name_pattern, height_range, max_angle)


def _synapse_build_tree(group_view, pattern):
    """Compatibility implementation of SynapseFormerView.build_tree()."""
    size = 0
    for container in group_view.containers.values():
        splines = container.object.data.splines
        for spline_index, spline in enumerate(splines):
            section_name = container.spline_index2section[spline_index].name
            if fnmatch(section_name, pattern):
                size += len(spline.bezier_points)

    tree = mathutils.kdtree.KDTree(size)
    node2terminal = {}
    max_radius = 0
    node_id = 0

    for container in group_view.containers.values():
        cell_obj = container.object
        splines = cell_obj.data.splines
        mw = cell_obj.matrix_world

        for spline_id, spline in enumerate(splines):
            section = container.spline_index2section[spline_id]
            section_name = section.name
            if not fnmatch(section_name, pattern):
                continue

            arc_lengths = section.arc_lengths()
            tot_length = arc_lengths[-1]

            for pt_id, pt in enumerate(spline.bezier_points):
                loc = Vector(_matmul(mw, pt.co)).copy().freeze()
                x = arc_lengths[pt_id] / tot_length
                seg_i = min(floor(section.nseg * x), section.nseg - 1)
                tree.insert(loc, node_id)
                node2terminal[node_id] = SynapseTerminal(loc, pt.radius, section_name, pt_id, x, seg_i)
                if pt.radius > max_radius:
                    max_radius = pt.radius
                node_id += 1

    tree.balance()
    return tree, node2terminal, max_radius


def _synapse_find_pairs(group_view1, view1_pattern, group_view2, group2_tree, group2_node2synterm,
                        max_dist, use_radii, max_radius, max_syns_per_pt):
    """Compatibility implementation of SynapseFormerView.find_pairs()."""
    pair_heap = []
    search_dist = max_dist + max_radius if use_radii else max_dist

    for container1 in group_view1.containers.values():
        cell_obj = container1.object
        mw = cell_obj.matrix_world

        for spline1_id, spline1 in enumerate(cell_obj.data.splines):
            section = container1.spline_index2section[spline1_id]
            section_name = section.name

            if not fnmatch(section_name, view1_pattern):
                continue

            arc_lengths = section.arc_lengths()
            tot_length = arc_lengths[-1]

            for pt1_id, pt1 in enumerate(spline1.bezier_points):
                pt_glob = Vector(_matmul(mw, pt1.co.copy())).freeze()
                matches = group2_tree.find_range(
                    pt_glob,
                    search_dist + (pt1.radius if use_radii else 0),
                )

                if len(matches) > 0:
                    x = arc_lengths[pt1_id] / tot_length
                    seg_i = min(floor(section.nseg * x), section.nseg - 1)

                for _pt2_glob, node2_id, dist in matches:
                    term1 = SynapseTerminal(pt_glob, pt1.radius, section_name, pt1_id, x, seg_i)
                    term2 = group2_node2synterm[node2_id]
                    pair = SynapsePair(term1, term2, dist)

                    if use_radii:
                        true_dist = dist - pt1.radius - term2.radius
                        if true_dist <= max_dist:
                            pair.length = true_dist
                            heappush(pair_heap, (true_dist, pair))
                    else:
                        heappush(pair_heap, (dist, pair))

    used_pt_counts = {}
    result_pairs = []

    while len(pair_heap) > 0:
        _dist, pair = heappop(pair_heap)
        pt1, pt2 = pair.source.loc, pair.dest.loc

        pt1_count = used_pt_counts.get(pt1, 0)
        pt2_count = used_pt_counts.get(pt2, 0)

        if pt1_count >= max_syns_per_pt or pt2_count >= max_syns_per_pt:
            continue

        used_pt_counts[pt1] = pt1_count + 1
        used_pt_counts[pt2] = pt2_count + 1
        result_pairs.append(pair)

    return result_pairs


def _ensure_blenderneuron_ready():
    """
    Register BlenderNEURON classes/properties and create a live Blender node.

    The upstream addon startup path assumes interactive UI startup and a modal
    operator lifecycle. For one-shot background slice exports we only need the
    registered operators/properties plus a connected ``BlenderNode`` instance.
    """
    scene = bpy.context.scene

    from blenderneuron.blender.blenderrootgroup import BlenderRootGroup
    from blenderneuron.blender.views.objectview import ObjectViewAbstract
    from blenderneuron.blender.views.curvecontainer import CurveContainer
    from blenderneuron.blender.views.synapseformerview import SynapseFormerView
    from blenderneuron.blender.views.vectorconfinerview import VectorConfinerView
    import blenderneuron.blender.utils as blenderneuron_utils

    if not getattr(BlenderRootGroup, "_obgpu_blender28_compat", False):
        def _default_color_get(self):
            material = self.color_ramp_material
            if material is None:
                return [1.0, 1.0, 1.0]
            if hasattr(material, "diffuse_ramp"):
                return material.diffuse_ramp.elements[0].color[0:3]
            return list(material.diffuse_color[0:3])

        def _default_color_set(self, value):
            material = self.color_ramp_material
            if material is None:
                return
            if hasattr(material, "diffuse_ramp"):
                material.diffuse_ramp.elements[0].color = list(value) + [1]
            else:
                material.diffuse_color = list(value) + [1]

        def _create_color_ramp_material(self, default_color):
            name = self.name + '_color_ramp'
            mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
            if hasattr(mat, "use_diffuse_ramp"):
                mat.use_diffuse_ramp = True
                mat.diffuse_ramp.elements[0].color = list(default_color) + [1]
                mat.diffuse_ramp.elements[-1].color = [1] * 4
            else:
                mat.diffuse_color = list(default_color) + [1]
            return name

        BlenderRootGroup.default_color = property(_default_color_get, _default_color_set)
        BlenderRootGroup.create_color_ramp_material = _create_color_ramp_material
        BlenderRootGroup._obgpu_blender28_compat = True

    if not getattr(ObjectViewAbstract, "_obgpu_blender28_compat", False):
        def _objectview_make_curve_template(self):
            curve_template = bpy.data.curves.new(self.group.name + "_bezier", type='CURVE')
            curve_template.dimensions = '3D'
            curve_template.resolution_u = self.group.segment_subdivisions
            curve_template.fill_mode = 'FULL'
            curve_template.bevel_depth = 0.0 if self.group.as_lines else 1.0
            curve_template.bevel_resolution = int((self.group.circular_subdivisions - 4) / 2.0)
            for attr in ("show_normal_face", "show_handles"):
                if hasattr(curve_template, attr):
                    setattr(curve_template, attr, False)
            self.curve_template_name = curve_template.name

        def _objectview_on_first_link(self):
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type != 'VIEW_3D':
                        continue
                    for space in area.spaces:
                        if space.type != 'VIEW_3D':
                            continue
                        if hasattr(space, "grid_scale"):
                            space.grid_scale = 100.0
                        elif hasattr(space, "overlay") and hasattr(space.overlay, "grid_scale"):
                            space.overlay.grid_scale = 100.0
                        if hasattr(space, "clip_end"):
                            space.clip_end = 99999
                        if hasattr(space, "show_relationship_lines"):
                            space.show_relationship_lines = False

            if not bpy.app.background:
                try:
                    lights = bpy.data.lights if hasattr(bpy.data, "lights") else bpy.data.lamps
                    sun_exists = any(light.type == 'SUN' for light in lights)
                    if not sun_exists:
                        if hasattr(bpy.ops.object, "light_add"):
                            bpy.ops.object.light_add(type="SUN", location=[500] * 3)
                        elif hasattr(bpy.ops.object, "lamp_add"):
                            bpy.ops.object.lamp_add(type="SUN", location=[500] * 3)
                except Exception:
                    pass

            for camera in bpy.data.cameras:
                camera.clip_end = 99999

        def _objectview_select_containers(self, select=True, pattern=None, pattern_inverse=False):
            bpy.ops.object.select_all(action='DESELECT')

            for container in self.containers.values():
                if pattern is not None:
                    matches = fnmatch(container.name, pattern)
                    if (not pattern_inverse and not matches) or (pattern_inverse and matches):
                        continue
                _select_object(container.get_object(), select)

        def _objectview_zoom_to_containers(self):
            if bpy.app.background:
                return

            self.select_containers(True)
            selected = _selected_objects()
            active_object = selected[0] if selected else None
            override = _view3d_context_override(active_object)
            if override is not None:
                try:
                    bpy.ops.view3d.view_selected(override, use_all_regions=False)
                except TypeError:
                    bpy.ops.view3d.view_selected(override)
            self.select_containers(False)

        def _objectview_containers_to_mesh(self):
            self.select_containers()
            selected = _selected_objects()
            if not selected:
                return
            active_ob = selected[0]
            _set_active_object(active_ob)
            override = _view3d_context_override(active_ob)
            if override is not None:
                bpy.ops.object.convert(override, target='MESH', keep_original=False)
            else:
                bpy.ops.object.convert(target='MESH', keep_original=False)

        def _objectview_mesh_containers_to_curves(self):
            self.select_containers()
            selected = _selected_objects()
            if not selected:
                return
            active_ob = selected[0]
            _set_active_object(active_ob)
            override = _view3d_context_override(active_ob)
            if override is not None:
                bpy.ops.object.convert(override, target='CURVE', keep_original=False)
            else:
                bpy.ops.object.convert(target='CURVE', keep_original=False)

        ObjectViewAbstract.make_curve_template = _objectview_make_curve_template
        ObjectViewAbstract.on_first_link = _objectview_on_first_link
        ObjectViewAbstract.select_containers = _objectview_select_containers
        ObjectViewAbstract.zoom_to_containers = _objectview_zoom_to_containers
        ObjectViewAbstract.containers_to_mesh = _objectview_containers_to_mesh
        ObjectViewAbstract.mesh_containers_to_curves = _objectview_mesh_containers_to_curves
        ObjectViewAbstract._obgpu_blender28_compat = True

    if not getattr(CurveContainer, "_obgpu_blender28_compat", False):
        @staticmethod
        def _curvecontainer_create_material(name, color, brightness):
            mat = bpy.data.materials.new(name)
            rgba = list(color) + ([1.0] if len(color) == 3 else [])
            if hasattr(mat, "diffuse_color"):
                mat.diffuse_color = rgba
            mat.use_nodes = True

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            links.clear()
            nodes.clear()

            out_node = nodes.new('ShaderNodeOutputMaterial')
            emit_node = nodes.new('ShaderNodeEmission')
            emit_node.location = [-200, 0]
            emit_node.inputs['Strength'].default_value = brightness
            emit_node.inputs['Color'].default_value = rgba
            links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])

            return mat

        def _curvecontainer_link(self):
            _scene_link_object(self.get_object())
            self.linked = True

        def _curvecontainer_unlink(self):
            try:
                _scene_unlink_object(self.get_object())
            except RuntimeError:
                pass
            self.linked = False

        CurveContainer.create_material = _curvecontainer_create_material
        CurveContainer.link = _curvecontainer_link
        CurveContainer.unlink = _curvecontainer_unlink
        CurveContainer._obgpu_blender28_compat = True

    if not getattr(VectorConfinerView, "_obgpu_blender28_compat", False):
        @staticmethod
        def _vectorconfiner_closest_point_on_object(global_pt, mesh_obj):
            return SliceBuilderBlender.closest_point_on_object(np.array(global_pt), mesh_obj)

        @staticmethod
        def _vectorconfiner_align_object_towards(ob, pt_global, max_angle):
            max_angle = max_angle / 180.0 * 3.141592

            ob_mw = ob.matrix_world
            end = ob.data.splines[0].bezier_points[-1].co
            start = ob.data.splines[0].bezier_points[0].co
            desired = _matmul(ob_mw.inverted(), pt_global)
            v_start = end - start
            v_des = desired - start
            q = v_start.rotation_difference(v_des).to_euler()
            q = Euler(list(map(lambda angle: min(max(angle, -max_angle), max_angle), q))).to_quaternion()

            ob.matrix_basis = _matmul(ob.matrix_basis.copy(), q.to_matrix().to_4x4())

            if ob.parent is None:
                ob.matrix_world = ob.matrix_basis
            else:
                ob.matrix_world = _matmul(_matmul(ob.parent.matrix_world, ob.matrix_parent_inverse), ob.matrix_basis)

            _scene_update()

        @staticmethod
        def _vectorconfiner_confine_between_meshes(obj, start_mesh, end_mesh, height_low, height_high, max_angle, iters=11):
            height_fraction = height_low + (height_high - height_low) * random.random()
            sec_start_loc = _matmul(obj.matrix_world, obj.data.splines[0].bezier_points[0].co)

            for _ in range(iters):
                tip_loc = _matmul(obj.matrix_world, obj.data.splines[0].bezier_points[-1].co)
                vec_sec_dir = (tip_loc - sec_start_loc).normalized()
                del vec_sec_dir

                closest_on_start, _dist_to_start = VectorConfinerView.closest_point_on_object(tip_loc, start_mesh)
                closest_on_end, _dist_to_end = VectorConfinerView.closest_point_on_object(tip_loc, end_mesh)

                closest_on_start = Vector(closest_on_start)
                closest_on_end = Vector(closest_on_end)

                vec_start2end = (closest_on_end - closest_on_start).normalized()
                vec_start2tip = (tip_loc - closest_on_start).normalized()
                angle = acos(min(max(vec_start2end.dot(vec_start2tip), -1), 1)) * 180 / pi

                above = angle < 90 - 0.02
                if not above:
                    vec_start2tip *= -1

                height = (closest_on_end - closest_on_start).length
                align_target = closest_on_start + vec_start2end * height * height_fraction
                VectorConfinerView.align_object_towards(obj, align_target, max_angle / iters)

        VectorConfinerView.closest_point_on_object = _vectorconfiner_closest_point_on_object
        VectorConfinerView.align_object_towards = _vectorconfiner_align_object_towards
        VectorConfinerView.confine_between_meshes = _vectorconfiner_confine_between_meshes
        VectorConfinerView._obgpu_blender28_compat = True

    if not getattr(SynapseFormerView, "_obgpu_blender28_compat", False):
        def _synapseformer_make_curve(self):
            curve = bpy.data.curves.new("SynapsePreviewCurve", type='CURVE')
            curve.dimensions = '3D'
            curve.resolution_u = 0
            curve.fill_mode = 'FULL'
            curve.bevel_depth = 0.0
            for attr in ("show_normal_face", "show_handles"):
                if hasattr(curve, attr):
                    setattr(curve, attr, False)
            return curve

        @staticmethod
        def _synapseformer_build_tree(group_view, pattern):
            size = 0
            for container in group_view.containers.values():
                splines = container.object.data.splines
                for spline_index, spline in enumerate(splines):
                    section_name = container.spline_index2section[spline_index].name
                    if fnmatch(section_name, pattern):
                        size += len(spline.bezier_points)

            tree = mathutils.kdtree.KDTree(size)
            node2terminal = {}
            max_radius = 0
            node_id = 0

            for container in group_view.containers.values():
                cell_obj = container.object
                splines = cell_obj.data.splines
                mw = cell_obj.matrix_world

                for spline_id, spline in enumerate(splines):
                    section = container.spline_index2section[spline_id]
                    section_name = section.name
                    if not fnmatch(section_name, pattern):
                        continue

                    arc_lengths = section.arc_lengths()
                    tot_length = arc_lengths[-1]

                    for pt_id, pt in enumerate(spline.bezier_points):
                        loc = Vector(_matmul(mw, pt.co)).copy().freeze()
                        x = arc_lengths[pt_id] / tot_length
                        seg_i = min(floor(section.nseg * x), section.nseg - 1)
                        tree.insert(loc, node_id)
                        node2terminal[node_id] = SynapseTerminal(loc, pt.radius, section_name, pt_id, x, seg_i)
                        radius = pt.radius
                        if radius > max_radius:
                            max_radius = radius
                        node_id += 1

            tree.balance()
            return tree, node2terminal, max_radius

        @staticmethod
        def _synapseformer_find_pairs(group_view1, view1_pattern, group_view2, group2_tree, group2_node2synterm,
                                      max_dist, use_radii, max_radius, max_syns_per_pt):
            pair_heap = []
            search_dist = max_dist + max_radius if use_radii else max_dist

            for container1 in group_view1.containers.values():
                cell_obj = container1.object
                mw = cell_obj.matrix_world

                for spline1_id, spline1 in enumerate(cell_obj.data.splines):
                    section = container1.spline_index2section[spline1_id]
                    section_name = section.name
                    if not fnmatch(section_name, view1_pattern):
                        continue

                    arc_lengths = section.arc_lengths()
                    tot_length = arc_lengths[-1]

                    for pt1_id, pt1 in enumerate(spline1.bezier_points):
                        pt_glob = Vector(_matmul(mw, pt1.co.copy())).freeze()
                        matches = group2_tree.find_range(
                            pt_glob,
                            search_dist + (pt1.radius if use_radii else 0),
                        )

                        if len(matches) > 0:
                            x = arc_lengths[pt1_id] / tot_length
                            seg_i = min(floor(section.nseg * x), section.nseg - 1)

                        for _pt2_glob, node2_id, dist in matches:
                            term1 = SynapseTerminal(pt_glob, pt1.radius, section_name, pt1_id, x, seg_i)
                            term2 = group2_node2synterm[node2_id]
                            pair = SynapsePair(term1, term2, dist)

                            if use_radii:
                                true_dist = dist - pt1.radius - term2.radius
                                if true_dist <= max_dist:
                                    pair.length = true_dist
                                    heappush(pair_heap, (true_dist, pair))
                            else:
                                heappush(pair_heap, (dist, pair))

            used_pt_counts = {}
            result_pairs = []

            while len(pair_heap) > 0:
                _dist, pair = heappop(pair_heap)
                pt1, pt2 = pair.source.loc, pair.dest.loc

                pt1_count = used_pt_counts.get(pt1, 0)
                pt2_count = used_pt_counts.get(pt2, 0)

                if pt1_count >= max_syns_per_pt or pt2_count >= max_syns_per_pt:
                    continue

                used_pt_counts[pt1] = pt1_count + 1
                used_pt_counts[pt2] = pt2_count + 1
                result_pairs.append(pair)

            return result_pairs

        def _synapseformer_get_synapse_locations(self, max_dist, use_radii, max_syns_per_pt,
                                                 section_pattern_source, section_pattern_dest):
            dest_group_tree, dest_node2synterm, dest_max_radius = self.build_tree(self.dest_group.view, section_pattern_dest)
            syn_pairs = self.find_pairs(
                self,
                section_pattern_source,
                self.dest_group.view,
                dest_group_tree,
                dest_node2synterm,
                max_dist,
                use_radii,
                dest_max_radius,
                max_syns_per_pt,
            )

            bez = bpy.data.objects.new("SynapsePreview", self.make_curve())
            for pair in syn_pairs:
                spline = bez.data.splines.new('BEZIER')
                bez_pts = spline.bezier_points
                bez_pts.add(1)
                bez_pts[0].co = pair.source.loc
                bez_pts[1].co = pair.dest.loc

            bpy.ops.object.select_all(action='DESELECT')
            _scene_link_object(bez)
            _scene_update()
            _select_object(bez, True)
            _set_active_object(bez)

            override = _view3d_context_override(bez)
            if override is not None:
                bpy.ops.object.mode_set(override, mode='EDIT')
                bpy.ops.curve.select_all(override, action='SELECT')
                bpy.ops.curve.handle_type_set(override, type='AUTOMATIC')
                bpy.ops.object.mode_set(override, mode='OBJECT')

            self.synapse_container_name = bez.name
            self.synapse_pairs = syn_pairs
            return syn_pairs

        SynapseFormerView.make_curve = _synapseformer_make_curve
        SynapseFormerView.build_tree = _synapseformer_build_tree
        SynapseFormerView.find_pairs = _synapseformer_find_pairs
        SynapseFormerView.get_synapse_locations = _synapseformer_get_synapse_locations
        SynapseFormerView._obgpu_blender28_compat = True

    if not getattr(blenderneuron_utils, "_obgpu_blender28_compat", False):
        blenderneuron_utils.get_operator_context_override = _view3d_context_override
        blenderneuron_utils._obgpu_blender28_compat = True

    if not hasattr(bpy.types.Scene, "BlenderNEURON") or not hasattr(scene, "BlenderNEURON"):
        from blenderneuron.blender.utils import register_module_classes
        import blenderneuron.blender.operators.connection
        import blenderneuron.blender.panels.connection
        import blenderneuron.blender.properties.connection
        import blenderneuron.blender.operators.rootgroup
        import blenderneuron.blender.panels.rootgroup
        import blenderneuron.blender.properties.rootgroup

        register_module_classes(blenderneuron.blender.operators.rootgroup)
        register_module_classes(blenderneuron.blender.panels.rootgroup)
        register_module_classes(blenderneuron.blender.properties.rootgroup)
        blenderneuron.blender.properties.rootgroup.register()

        register_module_classes(blenderneuron.blender.operators.connection)
        register_module_classes(blenderneuron.blender.panels.connection)
        register_module_classes(blenderneuron.blender.properties.connection)
        blenderneuron.blender.properties.connection.register()

    node = getattr(bpy.types.Object, "BlenderNEURON_node", None)
    if node is None:
        from blenderneuron.blender.blendernode import BlenderNode

        def on_client_connected(comm_node):
            scene.BlenderNEURON.simulator_settings.from_neuron()

        node = BlenderNode(on_client_connected=on_client_connected)
        bpy.types.Object.BlenderNEURON_node = node
        scene.BlenderNEURON.clear()

        if node.client is not None:
            node.add_group()
            node.add_synapse_set()

    return node


def auto_start(*_args):
    """
    A Blender startup script that starts the SliceBuilder on Blender startup
    """

    # Remove auto-execute command after starting
    handler_list = _auto_start_handler_list()
    if auto_start in handler_list:
        handler_list.remove(auto_start)

    # Assuming starting at repo root
    sys.path.append(os.getcwd())

    _ensure_blenderneuron_ready()

    # Create a slice builder class
    sbb = bpy.types.Object.SliceBuilder = SliceBuilderBlender(**slice_builder_env_kwargs())

    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(sbb.add_mc)
    # lp.add_function(sbb.add_tc)
    # lp.add_function(sbb.add_gc)
    # lp.add_function(sbb.import_instance)
    # profiled_build = lp(sbb.build)
    # profiled_build()
    # lp.print_stats()

    sbb.build()



class SliceBuilderBlender:
    @property
    def node(self):
        """
        Returns the BlenderNEURON node that runs within Blender
        """

        return bpy.types.Object.BlenderNEURON_node

    @property
    def neuron(self):
        """
        Returns the client class that communicates with NEURON via methods defined in `olfactorybulb.slicebuilder.nrn`
        """
        return self.node.client

    @property
    def slice_dir(self):
        """
        Returns the path to the directory where virtual slice cell and synapse files will be saved
        """
        slice_dir = os.path.abspath(os.path.dirname(slices.__file__))
        return os.path.join(slice_dir, self.slice_name)

    def __init__(self,
                 odors=['Apple'],
                 slice_object_name='DorsalColumnSlice',
                 slice_output_name=None,
                 max_mcs=10, max_tcs=None, max_gcs=300,  # Uses mouse ratios if None
                 max_eplis=0,
                 mc_particles_object_name='2 ML Particles',
                 tc_particles_object_name='1 OPL Particles',
                 gc_particles_object_name='4 GRL Particles',
                 epli_particles_object_name=None,
                 glom_particles_object_name='0 GL Particles',
                 glom_layer_object_name='0 GL',
                 outer_opl_object_name='1 OPL-Outer',
                 inner_opl_object_name='1 OPL-Inner',
                 enable_epl_interneurons=False,
                 epl_interneuron_model=DEFAULT_EPLI_MODEL_KEY,
                 epl_interneuron_family=None,
                 epli_depth_min_fraction=0.2,
                 epli_depth_max_fraction=0.8):
        """
        Prepares the slice builder

        :param odors: A list of odors whose glomeruli to include (e.g. ['Apple', 'Mint']), use 'all' for all gloms.
        :param slice_object_name: The name of the Blender mesh that defines the shape of the virtual slice
        :param slice_output_name: Optional output directory name for the generated slice assets.
        :param max_mcs: The maximum number of MCs to include in the model
        :param max_tcs: Maximum number of TCs. Use None to use mouse MC-TC ratio.
        :param max_gcs: Maximum number of GCs. Use None to use mouse MC-GC ratio.
        :param max_eplis: Maximum number of optional EPL interneurons. 0 disables the population.
        :param mc_particles_object_name: The name of the Blender object that defines MC soma locations
        :param tc_particles_object_name: The name of the Blender object that defines TC soma locations
        :param gc_particles_object_name: The name of the Blender object that defines GC soma locations
        :param epli_particles_object_name: Optional Blender object that defines EPLI soma candidate locations.
        :param glom_particles_object_name: The name of the Blender object that defines glomerulus locations
        :param glom_layer_object_name: The name of the Blender object that defines the geometry of glomerular layer
        :param outer_opl_object_name: The name of the Blender object that defines the outer boundary of the OPL layer
        :param inner_opl_object_name: The name of the Blender object that defines the inner boundary of the OPL layer
        """

        # In mouse, for each MC, there are:
        # See: model-data.sql > measurement table > mc/gc/tc_count entries for sources
        #  2.36 TCs
        if max_tcs is None:
            max_tcs = int(round(max_mcs * 2.36))

        #  16.97 GCs
        if max_gcs is None:
            max_gcs = int(round(max_mcs * 16.97))


        self.odors = odors
        self.glom_cells = {}

        self.slice_object_name = slice_object_name
        self.slice_name = slice_output_name or slice_object_name

        self.max_mcs = max_mcs
        self.max_tcs = max_tcs
        self.max_gcs = max_gcs
        self.max_eplis = max_eplis

        self.glom_particles_name = glom_particles_object_name
        self.tc_particles_name = tc_particles_object_name
        self.mc_particles_name = mc_particles_object_name
        self.gc_particles_name = gc_particles_object_name
        self.epli_particles_name = epli_particles_object_name or tc_particles_object_name

        self.glom_layer_object_name = glom_layer_object_name
        self.outer_opl_object_name = outer_opl_object_name
        self.inner_opl_object_name = inner_opl_object_name
        self.enable_epl_interneurons = epli_population_enabled(
            enable_epl_interneurons=enable_epl_interneurons,
            max_epl_interneurons=max_eplis,
        )
        self.epli_depth_min_fraction = float(epli_depth_min_fraction)
        self.epli_depth_max_fraction = float(epli_depth_max_fraction)
        self.epli_model_spec = (
            resolve_epli_model_spec(model=epl_interneuron_model, family=epl_interneuron_family)
            if self.enable_epl_interneurons
            else None
        )

        # Within slice
        self.get_cell_locations()

        self.get_cell_base_model_info()

        # Show as section objects
        self.create_groups()

        # Clear slice files
        self.clear_slice_files()

    def build(self, seed=0):
        """
        Positions MC/TC/GC models within the OB layers, identifies synapse locations, and saves the model
        for later simulation in NEURON

        :param seed: The random seed to use (to assist reproducibility)
        """

        random.seed(seed)

        self.max_alignment_angle = 35

        for i, loc in enumerate(self.mc_locs):
            print('Adding MC %s' % i)
            self.add_mc(loc)

        for i, loc in enumerate(self.tc_locs):
            print('Adding TC %s' % i)
            self.add_tc(loc)

        for i, loc in enumerate(self.gc_locs):
            print('Adding GC %s' % i)
            self.add_gc(loc)

        for i, loc in enumerate(getattr(self, "epli_locs", [])):
            print('Adding EPLI %s' % i)
            self.add_epli(loc)

        # Add synapse sets
        self.add_synapse_sets()

        # Select all cells in groups
        self.node.groups['MCs'].select_roots('All','mc*')
        self.node.groups['TCs'].select_roots('All','tc*')
        self.node.groups['GCs'].select_roots('All','gc*')
        if self.enable_epl_interneurons and EPLI_GROUP_NAME in self.node.groups:
            self.node.groups[EPLI_GROUP_NAME].select_roots('All', 'PVCRH*')

        connected_source_cells_by_group = {}

        # Find and save syns in all synapse sets
        for syn_set in self.node.synapse_sets:
            file = os.path.join(self.slice_dir, make_safe_filename(syn_set.name)+'.json')
            print('Saving synapse set "'+syn_set.name+'" saved to: ' + file)

            pairs = self.get_synapse_locations_for_set(syn_set)

            # Track connected source-side interneurons for optional pruning.
            for pair in pairs:
                source_cell = pair.source.section_name[:pair.source.section_name.find(']')+1]
                connected_source_cells_by_group.setdefault(syn_set.group_source, set()).add(source_cell)

            self.save_synapses_for_set(syn_set, file)

        # Remove unconnected source-side interneurons that would not contribute
        # to simulation output. Keep MCs/TCs untouched.
        for source_group_name in ('GCs', EPLI_GROUP_NAME):
            connected_cells = connected_source_cells_by_group.get(source_group_name)
            if source_group_name not in self.node.groups or connected_cells is None:
                continue
            self.node.groups[source_group_name].include_roots_by_name(
                [cell + '.soma' for cell in connected_cells],
                exclude_others=True
            )

        # Save all cells
        for group in self.node.groups.values():
            file = os.path.join(self.slice_dir, make_safe_filename(group.name)+'.json')
            print('Saving cell group %s %s to: %s'%(len(group.roots.keys()), group.name, file))
            group.to_file(file)

        # Save glom-cell associations
        file = os.path.join(self.slice_dir, 'glom_cells.json')
        with open(file, 'w') as f:
            print('Saving glomerulus-cells links to: ' + file)
            json.dump(self.glom_cells, f)

        # Initially, reduce group display detail levels - All can be changed in Blender GUI
        for group_name in default_slice_group_names(include_epli=self.enable_epl_interneurons):
            if group_name not in self.node.groups:
                continue
            self.node.groups[group_name].interaction_granularity = 'Cell'
            self.node.groups[group_name].as_lines = True

        # Show all group cells
        print('Creating blender scene...')
        if not bpy.app.background:
            bpy.ops.blenderneuron.display_groups()
        print('DONE')

    def add_synapse_sets(self):
        """
        Creates reciprocal synapse sets between configured slice populations.
        """

        # Delete the default set
        self.node.synapse_sets.remove(0)

        for blueprint in default_slice_synapse_blueprints(include_epli=self.enable_epl_interneurons):
            self.create_synapse_set(**blueprint)

    def create_synapse_set(
        self,
        group_from,
        group_to,
        max_distance=5,
        section_pattern_source="*apic*",
        section_pattern_dest="*dend*",
        synapse_name_dest='GabaSyn',
        synapse_params_dest=None,
        is_reciprocal=True,
        synapse_name_source='AmpaNmdaSyn',
        synapse_params_source=None,
        create_spines=False,
        spine_neck_diameter=0.2,
        spine_head_diameter=1,
        spine_name_prefix='Spine',
        conduction_velocity=1,
        initial_weight=1,
        threshold=0,
    ):
        """
        Defines a synapse set between two BlenderNEURON groups of cells

        :param group_from: Configured BlenderNEURON source group name
        :param group_to: Configured BlenderNEURON destination group name
        """

        new_set = self.node.add_synapse_set(group_from + '->' + group_to)
        new_set.group_source = group_from
        new_set.group_dest = group_to

        if synapse_params_dest is None:
            synapse_params_dest = {'gmax': 0.005, 'tau1': 1, 'tau2': 100}
        if synapse_params_source is None:
            synapse_params_source = {'gmax': 0.1}

        new_set.max_distance = max_distance
        new_set.use_radius = True
        new_set.max_syns_per_pt = 1
        new_set.section_pattern_source = section_pattern_source
        new_set.section_pattern_dest = section_pattern_dest
        new_set.synapse_name_dest = synapse_name_dest
        new_set.synapse_params_dest = str(synapse_params_dest)
        new_set.is_reciprocal = is_reciprocal
        new_set.synapse_name_source = synapse_name_source
        new_set.synapse_params_source = str(synapse_params_source)
        new_set.create_spines = create_spines
        new_set.spine_neck_diameter = spine_neck_diameter
        new_set.spine_head_diameter = spine_head_diameter
        new_set.spine_name_prefix = spine_name_prefix
        new_set.conduction_velocity = conduction_velocity
        new_set.initial_weight = initial_weight
        new_set.threshold = threshold

    def get_synapse_locations_for_set(self, syn_set):
        """
        Find synapse pairs for one synapse set using Blender 2.82-safe geometry code.
        """
        source_group = self.node.groups[syn_set.group_source]
        dest_group = self.node.groups[syn_set.group_dest]

        import_groups = [group for group in (source_group, dest_group) if group.state != 'imported']
        if len(import_groups) > 0:
            for group in import_groups:
                group.interaction_granularity = 'Cell'
                group.recording_granularity = 'Cell'
                group.record_activity = False
                group.import_synapses = False

            self.node.import_groups_from_neuron(import_groups)

        source_group.show(SynapseFormerView, dest_group)

        dest_group_tree, dest_node2synterm, dest_max_radius = _synapse_build_tree(
            source_group.view.dest_group.view,
            syn_set.section_pattern_dest,
        )

        pairs = _synapse_find_pairs(
            source_group.view,
            syn_set.section_pattern_source,
            source_group.view.dest_group.view,
            dest_group_tree,
            dest_node2synterm,
            syn_set.max_distance,
            syn_set.use_radius,
            dest_max_radius,
            syn_set.max_syns_per_pt,
        )
        source_group.view.synapse_pairs = pairs
        return pairs

    def save_synapses_for_set(self, syn_set, file_name):
        """
        Serialize one synapse set after ``get_synapse_locations_for_set`` populated the pair list.
        """
        source_group = self.node.groups[syn_set.group_source]
        if type(source_group.view) is not SynapseFormerView:
            raise RuntimeError("Synapse view is not initialized for save.")

        source_group.view.save_synapses(
            file_name,
            syn_set.name,
            syn_set.synapse_name_dest,
            syn_set.synapse_params_dest,
            syn_set.conduction_velocity,
            syn_set.synaptic_delay,
            syn_set.initial_weight,
            syn_set.threshold,
            syn_set.is_reciprocal,
            syn_set.synapse_name_source,
            syn_set.synapse_params_source,
            syn_set.create_spines,
            syn_set.spine_neck_diameter,
            syn_set.spine_head_diameter,
            syn_set.spine_name_prefix,
        )

    def clear_slice_files(self):
        """
        Deletes previously saved virtual slice .json files from the slice directory
        (e.g. olfactorybulb/slices/DorsalColumnSlice/.json)
        """

        dir = self.slice_dir
        os.makedirs(dir, exist_ok=True)

        # Match e.g. 'MCs.json'
        pattern = re.compile('.+json')

        for file in os.listdir(dir):
            if pattern.match(file) is not None:
                os.remove(os.path.abspath(os.path.join(dir, file)))

    def get_cell_locations(self):
        """
        Identifies the locations of glomeruli, mcs, tcs, and gcs that are contained by the virtual slice.
        """

        self.globalize_slice()

        odor_glom_ids = self.neuron.get_odor_gloms(self.odors)
        self.glom_locs = self.get_locs_within_slice(self.glom_particles_name, self.slice_object_name, odor_glom_ids)

        self.inner_opl_locs = self.get_opl_locs(self.inner_opl_object_name, self.slice_object_name)
        self.outer_opl_locs = self.get_opl_locs(self.outer_opl_object_name, self.slice_object_name)

        self.mc_locs = self.get_locs_within_slice(self.mc_particles_name, self.slice_object_name, limit=self.max_mcs)
        self.tc_locs = self.get_locs_within_slice(self.tc_particles_name, self.slice_object_name, limit=self.max_tcs)
        self.gc_locs = self.get_locs_within_slice(self.gc_particles_name, self.slice_object_name, limit=self.max_gcs)
        if self.enable_epl_interneurons:
            tc_particle_ids = {loc['id'] for loc in self.tc_locs} if self.epli_particles_name == self.tc_particles_name else set()
            self.epli_locs = self.get_epli_locs(
                self.epli_particles_name,
                exclude_particle_ids=tc_particle_ids,
                limit=self.max_eplis,
            )
        else:
            self.epli_locs = []

        print('Gloms:', len(self.glom_locs))
        print('TCs:', len(self.tc_locs))
        print('MCs:', len(self.mc_locs))
        print('GCs:', len(self.gc_locs))
        if self.enable_epl_interneurons:
            print('EPLIs:', len(self.epli_locs))

    def get_opl_locs(self, opl_name, slice_obj_name):
        """
        Gets the coordinates of inner or outer boundaries of the OPL layer that are contained by the virtual slice
        """

        obj = bpy.data.objects[opl_name]
        wm = obj.matrix_world

        locs = fast_get(obj.data.vertices, 'co', 3)

        def globalize(loc):
            return np.array(_matmul(wm, Vector(loc)))

        # Globalize the coordinates
        locs = np.array(list(map(globalize, locs)))

        slice_obj = bpy.data.objects[slice_obj_name]

        return np.array([pt
                         for pt in locs
                         if self.is_inside(Vector(pt), slice_obj)])

    def get_epli_locs(self, particle_obj_name, exclude_particle_ids=None, limit=None):
        """
        Select candidate soma locations for optional EPL interneurons.

        The first-pass implementation reuses one particle cloud and keeps only
        positions that sit within a configurable interior band of the EPL,
        estimated from distances to the inner and outer OPL boundaries.
        """

        exclude_particle_ids = set() if exclude_particle_ids is None else set(exclude_particle_ids)
        candidates = self.get_locs_within_slice(particle_obj_name, self.slice_object_name)
        result = []

        for candidate in candidates:
            if candidate['id'] in exclude_particle_ids:
                continue

            _closest_iopl_loc, dist_to_iopl, _dists_iopl = self.get_opl_distance_info(candidate['loc'], self.inner_opl_locs)
            _closest_oopl_loc, dist_to_oopl, _dists_oopl = self.get_opl_distance_info(candidate['loc'], self.outer_opl_locs)
            total_dist = dist_to_iopl + dist_to_oopl
            if total_dist <= 0:
                continue

            depth_fraction = dist_to_iopl / total_dist
            if self.epli_depth_min_fraction <= depth_fraction <= self.epli_depth_max_fraction:
                candidate = dict(candidate)
                candidate['depth_fraction'] = depth_fraction
                result.append(candidate)

        if limit is not None:
            print('Selecting %s/%s %s EPLI locations inside slice' % (limit, len(result), particle_obj_name))
            result = result[:limit]

        return result

    def get_cell_base_model_info(self):
        """
        Gets metadata info about each of the base (untransformed) MC, TC, and GC cell models
        """

        self.mc_base_models, self.tc_base_models, self.gc_base_models = \
            self.neuron.get_base_model_info()

        self.mc_base_models = OrderedDict(sorted(self.mc_base_models.items(), key=lambda i: i[0]))
        self.tc_base_models = OrderedDict(sorted(self.tc_base_models.items(), key=lambda i: i[0]))
        self.gc_base_models = OrderedDict(sorted(self.gc_base_models.items(), key=lambda i: i[0]))

        self.max_apic_mc_info = self.get_longest_apic_model(self.mc_base_models)
        self.mc_apic_lengths = self.get_apic_lengths(self.mc_base_models)

        self.max_apic_tc_info = self.get_longest_apic_model(self.tc_base_models)
        self.tc_apic_lengths = self.get_apic_lengths(self.tc_base_models)

        self.max_apic_gc_info = self.get_longest_apic_model(self.gc_base_models)
        self.gc_apic_lengths = self.get_apic_lengths(self.gc_base_models)

    @staticmethod
    def get_apic_lengths(base_models):
        """
        Gets the apical dendrite lengths of the specified base models

        :param base_models: List with metadata of base cell models
        :return: A numpy array of apical dendrite lengths
        """

        return np.array([c["apical_dendrite_reach"] for c in base_models.values()])

    def create_groups(self):
        """
        Creates empty BlenderNEURON cell groups for configured populations.
        """

        # Remove the default group
        self.node.groups['Group.000'].remove()

        # Create empty cell groups
        group_names = default_slice_group_names(include_epli=self.enable_epl_interneurons)
        groups = [self.node.add_group(name, False) for name in group_names]

        # show each section as blender objects - necessary for dend alignment
        for group in groups:
            group.interaction_granularity = 'Section'
            group.recording_granularity = 'Cell'
            group.record_activity = False

        # Add some color
        group_colors = default_slice_group_colors(include_epli=self.enable_epl_interneurons)
        for group in groups:
            group.default_color = group_colors[group.name]

    def globalize_slice(self):
        """
        Converts all points of the slice object to global coordinates (relative to scene origin)
        """

        # Apply all/any transformations to the slice
        slice = bpy.data.objects[self.slice_object_name]
        if hasattr(slice, "select_set"):
            slice.select_set(True)
            bpy.context.view_layer.objects.active = slice
        else:
            slice.select = True
            bpy.context.scene.objects.active = slice
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
        if hasattr(slice, "select_set"):
            slice.select_set(False)
        else:
            slice.select = False

    def add_mc(self, mc_pt):
        """
        Instantiates in NEURON, places, and orients a mitral cell within the mitral cell layer
        and confines its lateral dendrites to the curvature of the surrounding internal part
        of the outer plexiform layer.

        :param mc_pt: A dict whose 'loc' key contains a numpy array of xyz coordinates of the cell soma
        """

        # find the closest glom layer loc - cell will be pointed towards it
        closest_glom_loc, dist_to_gl = \
            self.closest_point_on_object(mc_pt['loc'], bpy.data.objects[self.glom_layer_object_name])

        longest_apic_reach = self.max_apic_mc_info["apical_dendrite_reach"]

        # Apics are too short use longest apic MC
        if dist_to_gl > longest_apic_reach:
            mc = self.max_apic_mc_info
        else:
            # get mcs with apics longer than dist to GL
            longer_idxs = np.where(self.mc_apic_lengths > dist_to_gl)[0]

            # pick a random mc from this list
            mc = self.get_random_model(self.mc_base_models, longer_idxs)

        # find a glom whose distance is as close to the length of the mc apic
        matching_glom_loc, matching_glom_id = self.find_matching_glom(mc_pt['loc'], mc)

        base_class = mc["class_name"]
        apic_glom_loc = matching_glom_loc

        # Create the selected MC in NRN
        instance_name = self.neuron.create_cell('MC', base_class)

        # Associate the cell with the glomerulus
        self.link_cell_to_glom(instance_name, matching_glom_id)

        # Import cell into Blender
        self.import_instance(instance_name, 'MCs')

        mc_soma, mc_apic_start, mc_apic_end = \
            self.get_key_mctc_section_objects(self.mc_base_models, base_class, instance_name)

        # Align apical towards the closest glom
        self.position_orient_align_mctc(mc_soma,
                                        mc_apic_start,
                                        mc_apic_end,
                                        mc_pt['loc'],
                                        closest_glom_loc,
                                        apic_glom_loc)

        # Retain the reoriented cell
        bpy.ops.blenderneuron.update_groups_with_view_data()

        self.confine_dends(
            'MCs',
            self.inner_opl_object_name,
            self.outer_opl_object_name,
            max_angle=self.max_alignment_angle,
            height_start=0,
            height_end=0.6
        )

        bpy.ops.blenderneuron.update_groups_with_view_data()

    def link_cell_to_glom(self, instance_name, matching_glom_id):
        """
        Adds a cell instance to a list of cells that belong to the specified glomerulus

        :param instance_name: The name of the cell as returned by NEURON
        :param matching_glom_id: The id of the glomerulus, with which to associate the cell
        """

        glom_cells = self.glom_cells.get(matching_glom_id, [])

        glom_cells.append(instance_name.replace('.soma',''))

        self.glom_cells[matching_glom_id] = glom_cells

    def import_instance(self, instance_name, group_name):
        """
        Imports and shows a cell instantiated in NEURON into Blender using a BlenderNEURON group

        :param instance_name: The name of the cell model to import (as named by NEURON)
        :param group_name: The name of the group to associate the cell with
        """

        # Get updated list of NRN cells in Blender
        bpy.ops.blenderneuron.get_cell_list_from_neuron()

        # Select the created instance
        group = self.node.groups[group_name]
        group.include_roots_by_name([instance_name], exclude_others=True)

        # Import group with the created cell and show it
        group.import_group()
        group.show()

    def add_tc(self, tc_pt):
        """
        Similar to `add_mc(pt)`. Instantiates, places, and orients a tufted cell model and
        confines its lateral dendrites to the outer portion of the outer plexiform layer.

        :param tc_pt: A dict with 'loc' that contains a numpy array of xyz coordinates of the TC soma
        """

        # find the closest glom layer loc - cell will be pointed towards it
        closest_glom_loc, dist_to_gl = \
            self.closest_point_on_object(tc_pt['loc'], bpy.data.objects[self.glom_layer_object_name])

        longest_apic_reach = self.max_apic_tc_info["apical_dendrite_reach"]

        # Apics are too short use longest apic TC
        if dist_to_gl > longest_apic_reach:
            tc = self.max_apic_tc_info
        else:
            # Apics are longer than distance
            # get tcs with apics longer than the closest glom,
            # but no further than ~200 um from glom (Source: Kikuta et. al. 2013)
            longer_idxs = np.where((self.tc_apic_lengths > dist_to_gl) &
                                   (self.tc_apic_lengths - dist_to_gl < 200))[0]

            # pick a random tc from this list
            tc = self.get_random_model(self.tc_base_models, longer_idxs)

        # find a glom whose distance is as close to the length of the tc apic
        matching_glom_loc, matching_glom_id = self.find_matching_glom(tc_pt['loc'], tc)

        base_class = tc["class_name"]
        apic_glom_loc = matching_glom_loc

        # Create the selected TC in NRN
        instance_name = self.neuron.create_cell('TC', base_class)

        # Associate the cell with the glomerulus
        self.link_cell_to_glom(instance_name, matching_glom_id)

        # Import it into Blender
        self.import_instance(instance_name, 'TCs')

        soma, apic_start, apic_end = \
            self.get_key_mctc_section_objects(self.tc_base_models, base_class, instance_name)

        # Align apical towards the closest glom
        self.position_orient_align_mctc(soma,
                                        apic_start,
                                        apic_end,
                                        tc_pt['loc'],
                                        closest_glom_loc,
                                        apic_glom_loc)

        # Retain the reoriented cell
        bpy.ops.blenderneuron.update_groups_with_view_data()

        self.confine_dends(
            'TCs',
            self.inner_opl_object_name,
            self.outer_opl_object_name,
            max_angle=self.max_alignment_angle,
            height_start=0.4,
            height_end=1.0
        )

        bpy.ops.blenderneuron.update_groups_with_view_data()

    def find_closest_glom(self, cell_loc):
        """
        Finds the closest glomerulus to the specified coordinates

        :param cell_loc: A numpy array of xyz coordinates
        :return: A tuple with the location of the closest glomerulus and distance to it
        """

        # Get distances to individual gloms
        glom_dists = self.dist_to_gloms(cell_loc)

        matching_glom_idx = np.argmin(glom_dists)
        matching_glom = self.glom_locs[matching_glom_idx]

        return matching_glom['loc'], glom_dists[matching_glom_idx]

    def find_matching_glom(self, cell_loc, cell_model_info):
        """
        Finds a glomerulus that is approximately the same distance from the soma
        as the length of the cells apical dendrite

        :param cell_loc: A numpy xyz array with coordinates
        :param cell_model_info: Cell model metadata dict with 'apical_dendrite_reach' key that contains the apical dendrite length
        :return: The xyz location and the id of the matching glomerulus
        """

        # Get distances to individual gloms
        glom_dists = self.dist_to_gloms(cell_loc)

        matching_glom_idx = np.argmin(np.abs(glom_dists - cell_model_info["apical_dendrite_reach"]))
        matching_glom = self.glom_locs[matching_glom_idx]

        return matching_glom['loc'], matching_glom['id']

    def get_opl_distance_info(self, cell_loc, pts):
        """
        Computes distances to and the closest point of an outer plexiform layer to a given point

        :param cell_loc: The xyz location of a point
        :param pts: The list of xyz coordinates of the OPL layer mesh
        :return: A tuple with closest location on the layer, the distance to that point, and a list of distances to all OPL points
        """

        dists = self.dist_to(pts, cell_loc)
        closest_idxs = np.argsort(dists)

        closest_loc = pts[closest_idxs][0]
        closest_dist = dists[closest_idxs][0]

        return closest_loc, closest_dist, dists

    @staticmethod
    def closest_point_on_object(global_pt, mesh_obj):
        """
        Gets the closest point and distance of a mesh object from a specified point

        :param global_pt: The point from which to measure distance
        :param mesh_obj: A mesh object
        :return: A tuple with an xyz numpy array of the closest point and distance to it
        """
        depsgraph = None
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            pass
        mesh_obj = _evaluated_mesh_obj(mesh_obj)

        local_pt = _matmul(mesh_obj.matrix_world.inverted(), Vector(global_pt))

        try:
            try:
                if depsgraph is not None:
                    _, mesh_pt, _, _ = mesh_obj.closest_point_on_mesh(local_pt, depsgraph=depsgraph)
                else:
                    _, mesh_pt, _, _ = mesh_obj.closest_point_on_mesh(local_pt)
            except TypeError:
                _, mesh_pt, _, _ = mesh_obj.closest_point_on_mesh(local_pt)

            mesh_pt_global = _matmul(mesh_obj.matrix_world, mesh_pt)
            dist = Vector(global_pt - mesh_pt_global).length
            return np.array(mesh_pt_global), dist
        except RuntimeError:
            pts = _object_vertex_points_global(mesh_obj)
            if len(pts) == 0:
                raise
            dists = np.sqrt(np.sum(np.square(pts - np.asarray(global_pt, dtype=float)), axis=1))
            idx = int(np.argmin(dists))
            return pts[idx], float(dists[idx])

    def add_gc(self, gc_pt):
        """
        Instantiates, places, and orients a granule cell in the granule cell layer

        :param gc_pt: XYZ array of GC soma location
        """

        # find the closest glom layer loc - cell will be pointed towards it
        glom_loc, glom_dist = \
            self.closest_point_on_object(gc_pt['loc'], bpy.data.objects[self.glom_layer_object_name])
        # glom_loc, glom_dist = self.find_closest_glom(gc_pt['loc'])

        # find the closest inner opl loc - apics must go beyond this
        closest_iopl_loc, dist_to_iopl, _dists_iopl = \
            self.get_opl_distance_info(gc_pt['loc'], self.inner_opl_locs)

        # find the closest outer opl loc - apics must stay under this
        closest_oopl_loc, dist_to_oopl, _dists_oopl = \
            self.get_opl_distance_info(gc_pt['loc'], self.outer_opl_locs)

        # Find such GC models whose apics are confined to the OPL
        # Specifically:
        # Get gcs with apics longer than the closest opl
        # AND
        # the apic does not exceed outer opl (an external margin is allowed)
        min_length = dist_to_iopl
        max_length = dist_to_oopl + 30

        matching_idxs = np.where((self.gc_apic_lengths > min_length) &
                                 (self.gc_apic_lengths < max_length))[0]

        # If no cell can be confined to OPL, leave that location blank
        if len(matching_idxs) == 0:
            return

        # pick a random gc from this list
        gc = self.get_random_model(self.gc_base_models, matching_idxs)

        base_class = gc["class_name"]
        apic_target_loc = glom_loc

        # Create the selected GC in NRN
        instance_name = self.neuron.create_cell('GC', base_class)

        # Import it into Blender
        self.import_instance(instance_name, 'GCs')

        soma, apic_start, apic_end = \
            self.get_key_mctc_section_objects(self.gc_base_models, base_class, instance_name)

        self.position_orient_cell(soma, apic_end, gc_pt['loc'], apic_target_loc)

        # Retain the reoriented cell
        bpy.ops.blenderneuron.update_groups_with_view_data()

    def add_epli(self, epli_pt):
        """
        Instantiate and place one optional fast EPL interneuron surrogate.

        The first-pass geometry is deliberately conservative: soma placed within
        a mid-EPL band, random in-plane rotation, then dendrites confined to the
        EPL corridor between the inner and outer OPL surfaces.
        """

        if self.epli_model_spec is None:
            raise RuntimeError("EPLI model spec is not configured.")

        instance_name = self.neuron.create_cell('EPLI', self.epli_model_spec.key)
        self.import_instance(instance_name, EPLI_GROUP_NAME)

        soma = bpy.data.objects[instance_name]
        soma.rotation_euler[2] = random.randrange(360) / 180.0 * pi
        soma.location = epli_pt['loc']
        _scene_update()

        bpy.ops.blenderneuron.update_groups_with_view_data()
        self.confine_dends(
            EPLI_GROUP_NAME,
            self.inner_opl_object_name,
            self.outer_opl_object_name,
            max_angle=self.max_alignment_angle,
            height_start=self.epli_depth_min_fraction,
            height_end=self.epli_depth_max_fraction,
        )
        bpy.ops.blenderneuron.update_groups_with_view_data()

    def get_random_model(self, base_models, longer_idxs):
        """
        Given a list of indices stored in longer_idxs, selects a random element of base_models

        :param base_models: A list of base cell models
        :param longer_idxs: A list of indices from which to pick a random index
        :return: A random cell model
        """

        rand_idx = longer_idxs[random.randrange(len(longer_idxs))]
        cell = list(base_models.values())[rand_idx]
        return cell

    def confine_dends(self, group_name, start_layer_name, end_layer_name, max_angle, height_start, height_end):
        """
        Confines the lateral dendrits of cells in a group to follow the curvature between two layers

         Height_start and height_end specify fractions that define the corridor between the layers to which
         the dendrites should be confined. E.g. To confine the dendrites in the halfway between the two layers,
         closer to the start layer, set height_start and height_end to 0 and 0.5. Set them to 0.5 and 1.0 to confine
         to the halfway that is closer to the end region. 1.0 corresponds to the local distance between the two layers.

         The two layers should be 'locally' parallel. E.g. two planes, or two concentric spheres. In OB model case,
         the OB layers are complex shapes, but concentric-like.

        :param group_name: The name of the group of cells to confine
        :param start_layer_name: The name of the first confinement layer
        :param end_layer_name: The name of the second confinement layer
        :param max_angle: The maximum angle that a dendritic branch can rotate to be confined between layers
        :param height_start: Fraction between 0-1
        :param height_end: Fraction between 0-1
        """

        group = self.node.groups[group_name]

        # Set the layers
        group.set_confiner_layers(start_layer_name, end_layer_name, max_angle, height_start, height_end)
        group.setup_confiner()
        settings = group.ui_group.layer_confiner_settings

        for root in group.roots.values():
            container_name = root.split_sections[0].name if root.was_split else root.name
            container = group.view.containers.get(container_name)
            if container is None:
                continue
            _confine_curve(
                container.object,
                settings.start_mesh,
                settings.end_mesh,
                settings.moveable_sections_pattern,
                [settings.height_min, settings.height_max],
                settings.max_bend_angle,
            )

    def save_transform(self, group_name, instance_name):
        """
        Saves trnsformed cells in a cell group to a NEURON compatible file

        :param group_name: The name of the BlenderNEURON cell group to save
        :param instance_name: The name of the file to save. Can be the name of the cell soma segment.
        """


        group = self.node.groups[group_name]

        # Make instance name a valid python module name (eg: MC1[0].soma -> MC1_0.py)
        file_name = instance_name \
                        .replace("].soma", "") \
                        .replace("[", "_") + \
                        ".py"

        # Save to slice folder
        path = os.path.join(self.slice_dir, file_name)

        # Save cells part of the group as files
        group.to_file(path)

    def get_key_mctc_section_objects(self, base_model_dict, base_class, instance_name):
        """
        Gets the blender objects of soma, apical dendrite start (base), and apical dendrite end (tuft) sections

        :param base_model_dict: A dicttionary of base cell model metadata info
        :param base_class: The name of a base cell class
        :param instance_name: The name of blender object of cell soma section
        :return: Blender objects of soma, apical dendrite start, and apical dendrite end
        """

        cell_info = base_model_dict[base_class]
        apic_pattern = instance_name.replace(".soma", "") + '.apic[%s]'

        bpy_objects = bpy.data.objects
        soma = bpy_objects[instance_name]

        apic_start = bpy_objects.get(apic_pattern % cell_info["apical_dendrite_start"])
        apic_end = bpy_objects.get(apic_pattern % cell_info["apical_dendrite_end"])

        return soma, apic_start, apic_end

    def get_longest_apic_model(self, base_model_dict):
        """
        Returns the metadata of the base cell model that has that longest apical dendrite

        :param base_model_dict: The dict of base cell model metadatas
        :return: A metadata object
        """

        cell_names, apic_lengths = zip(*[(cell["class_name"], cell["apical_dendrite_reach"])
                                         for cell in base_model_dict.values()])

        max_apic_idx = np.argmax(apic_lengths)

        return base_model_dict[cell_names[max_apic_idx]]

    def dist_to_gloms(self, loc):
        """
        Returns an array of distances to glomeruli from a given location

        :param loc: XYZ coordinate
        :return: Distances to glomeruli
        """

        return self.dist_to(np.array([glom['loc'] for glom in self.glom_locs]), loc)

    @staticmethod
    def dist_to(targets_array, loc):
        """
        Returns an array of distances to the specified list of points from a given point

        :param targets_array: The array of xzy target coordinates
        :param loc: The target coordinate
        :return: An array of distances
        """

        return np.sqrt(np.sum(np.square(targets_array - loc), axis=1))

    def get_locs_within_slice(self, particle_obj_name, slice_obj_name, allowed_particles=None, limit=None):
        """
        Gets a list of particle locations that are contained by a slice object

        :param particle_obj_name: A blender particle object
        :param slice_obj_name: A blender mesh object that represents the virtual slice
        :param allowed_particles: A list of particle ids that are allowed to be included. If None, not restricted.
        :param limit: A list of dicts with ids and locs of matching points
        :return:
        """

        particles_obj = bpy.data.objects[particle_obj_name]
        particles = particles_obj.particle_systems[0].particles
        particles_wm = particles_obj.matrix_world
        slice_obj = bpy.data.objects[slice_obj_name]

        result = [{ 'id': pid, 'loc': np.array(_matmul(particles_wm, ptc.location))}
                           for pid, ptc in enumerate(particles)
                           if (allowed_particles is None or pid in allowed_particles) and
                           self.is_inside(_matmul(particles_wm, ptc.location), slice_obj)]

        if limit is not None:
            print('Selecting %s/%s %s locations inside slice'%(limit, len(result), particle_obj_name))
            result = result[:limit]

        return result

    @staticmethod
    def is_inside(target_pt_global, mesh_obj, tolerance=1):
        """
        Determines if a target point is inside a mesh

        :param target_pt_global: Target xyz point, in global coordinates
        :param mesh_obj: Target mesh object
        :param tolerance: A tolerance in degrees to account for rounding error in detecting points inside. <=1 is generally sufficient.
        :return: True or False
        """
        depsgraph = None
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            pass
        mesh_obj = _evaluated_mesh_obj(mesh_obj)

        # Convert the point from global space to mesh local space
        target_pt_local = _matmul(mesh_obj.matrix_world.inverted(), target_pt_global)

        # Find the nearest point on the mesh and the nearest face normal
        try:
            if depsgraph is not None:
                _, pt_closest, face_normal, _ = mesh_obj.closest_point_on_mesh(target_pt_local, depsgraph=depsgraph)
            else:
                _, pt_closest, face_normal, _ = mesh_obj.closest_point_on_mesh(target_pt_local)
        except TypeError:
            _, pt_closest, face_normal, _ = mesh_obj.closest_point_on_mesh(target_pt_local)

        # Get the target-closest pt vector
        target_closest_pt_vec = (pt_closest - target_pt_local).normalized()

        # Compute the dot product = |a||b|*cos(angle)
        dot_prod = target_closest_pt_vec.dot(face_normal)

        # Get the angle between the normal and the target-closest-pt vector (from the dot prod)
        angle = acos(min(max(dot_prod, -1), 1)) * 180 / pi

        # Allow for some rounding error
        inside = angle < 90 - tolerance

        return inside

    def unparent(self, obj):
        """
        Removes the parent of a Blender object, and keeps the object in the same location

        :param obj: The blender object to unparent
        :return: The parent of the target object
        """

        prev_parent = obj.parent
        parented_wm = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = parented_wm
        return prev_parent

    def parent(self, obj, parent):
        """
        Make one blender object a parent of another

        :param obj: The child object
        :param parent: The child's parent object
        """

        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()

    def position_orient_align_mctc(self, soma, apic_start, apic_end, loc, closest_glom_loc, apic_glom_loc):
        """
        TODO: this appears to have the same/similar function as self.position_orient_cell. This may be
        redundant code.

        :param soma:
        :param apic_start:
        :param apic_end:
        :param loc:
        :param closest_glom_loc:
        :param apic_glom_loc:
        :return:
        """

        # Position and 'point' cell towards closest glom
        self.position_orient_cell(soma, apic_end, loc, closest_glom_loc)

        # Temporarily unparent the apic start (location becomes global)
        apic_start_parent = self.unparent(apic_start)
        apic_end_parent = self.unparent(apic_end)

        # Compute the start and end alignment vectors (start apic->end apic TO start apic->glom)
        # Relative to apic_start
        apic_start_wmi = apic_start.matrix_world.inverted()
        apic_end_loc = _matmul(apic_start_wmi, apic_end.location)

        startVec = Vector(apic_end_loc)
        endVec = Vector(_matmul(apic_start_wmi, Vector(apic_glom_loc)))

        # Reparent the apic end (so it rotates with the start apic)
        self.parent(apic_end, apic_end_parent)

        # Compute rotation quaternion and rotate the start apic by it
        initMW = apic_start.matrix_world.copy()
        rotM = startVec.rotation_difference(endVec).to_matrix().to_4x4()
        apic_start.matrix_world = _matmul(initMW, rotM)

        # Reparent the apic end (so it rotates with the start apic)
        self.parent(apic_start, apic_start_parent)

        _scene_update()

    def position_orient_cell(self, soma, apic_end, soma_loc, closest_glom_loc):
        """
        Position cell at a location, rotate it around its apical dendrite axis,
        and rotate it around the soma so it's apical dendrite 'points' towards
        the closest glomerulus location.

        :param soma: Blender object that holds the soma section
        :param apic_end: Blender object that holds the apical dendrite (furthest apical section on/near apical axis)
        :param soma_loc: The desired cell soma location
        :param closest_glom_loc: The location of the closest glomerulus
        """

        # Add random rotation around the apical axis
        soma.rotation_euler[2] = random.randrange(360) / 180.0 * pi

        # Position the soma
        soma.location = soma_loc

        # Update child matrices
        _scene_update()

        # Align the soma to be orthogonal to the soma-closest glom vector
        soma_wmi = soma.matrix_world.inverted()
        apic_end_world = _matmul(apic_end.matrix_world, apic_end.location)
        apic_end_loc = Vector(_matmul(soma_wmi, apic_end_world))
        glom_loc = Vector(_matmul(soma_wmi, Vector(closest_glom_loc)))

        initMW = soma.matrix_world.copy()
        rotM = apic_end_loc.rotation_difference(glom_loc).to_matrix().to_4x4()
        soma.matrix_world = _matmul(initMW, rotM)

        # Update child matrices
        _scene_update()

    def extend_apic(self, apic_start, apic_end, apic_glom_loc):
        """
        TODO: This is probably unused, leftover from an earlier version.

        :param apic_start:
        :param apic_end:
        :param apic_glom_loc:
        :return:
        """

        # Relative to apic_start
        glom_loc = _matmul(apic_start.matrix_world.inverted(), Vector(apic_glom_loc))
        apic_end_world = _matmul(apic_end.matrix_world, apic_end.location)
        apic_end_loc = _matmul(apic_start.matrix_world.inverted(), apic_end_world)

        apic_glom_diff = Vector(glom_loc - apic_end_loc)

        apic_start.location = apic_start.location.copy() + apic_glom_diff



# This makes it so the slice builder automatically runs when Blender loads
if bpy.app.background:
    auto_start()
else:
    _auto_start_handler_list().append(auto_start)
