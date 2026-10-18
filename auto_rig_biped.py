bl_info = {
    "name": "Auto Rig Biped (Mixamo-style)",
    "author": "john",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > AutoRig",
    "description": "One-click humanoid rig: places Mixamo-style joint markers, builds a "
                   "Mixamo-named armature (with fingers) and binds the mesh with automatic weights",
    "category": "Rigging",
}

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, PointerProperty
from mathutils import Vector

MARKER_PREFIX = "AR_"
MARKER_COLLECTION = "AutoRig Markers"
MARKERS = ["chin", "groin", "wrist_L", "wrist_R", "elbow_L", "elbow_R", "knee_L", "knee_R"]

FRONT_VEC = Vector((0.0, -1.0, 0.0))  # roll reference: bone Z points to the character's front

# --------------------------------------------------------------------------- naming

_MIXAMO = {
    "hips": "Hips", "spine": "Spine", "spine1": "Spine1", "spine2": "Spine2",
    "neck": "Neck", "head": "Head",
    "shoulder": "{S}Shoulder", "arm": "{S}Arm", "forearm": "{S}ForeArm", "hand": "{S}Hand",
    "upleg": "{S}UpLeg", "leg": "{S}Leg", "foot": "{S}Foot", "toe": "{S}ToeBase",
}
_BLENDER = {
    "hips": "hips", "spine": "spine", "spine1": "chest", "spine2": "upper_chest",
    "neck": "neck", "head": "head",
    "shoulder": "shoulder", "arm": "upper_arm", "forearm": "forearm", "hand": "hand",
    "upleg": "thigh", "leg": "shin", "foot": "foot", "toe": "toe",
}

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def bone_name(style, key, side=None):
    """key is a body key ('arm', 'hips', ...) or a finger key like 'Index2'."""
    finger = key.rstrip("0123456789") in FINGER_NAMES
    if style == "MIXAMO":
        s = {"L": "Left", "R": "Right"}.get(side, "")
        base = f"{s}Hand{key}" if finger else _MIXAMO[key].format(S=s)
        return "mixamorig:" + base
    if finger:
        base = f"{key[:-1].lower()}.{int(key[-1]):02d}"
    else:
        base = _BLENDER[key]
    return f"{base}.{side}" if side else base


# --------------------------------------------------------------------------- mesh analysis

def world_vertices(obj, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def front_sign(facing):
    return -1.0 if facing == "NEG_Y" else 1.0


def bounds(co):
    mn, mx = co.min(0), co.max(0)
    return mn[2], mx[2] - mn[2], mx[2], (mn[0] + mx[0]) / 2


def guess_markers(co, facing):
    """Heuristic joint positions from body proportions + extremities. Assumes T- or A-pose."""
    fs = front_sign(facing)
    lx = -fs  # character's left is +X when facing -Y
    z0, H, _, cx = bounds(co)
    V = Vector

    def band(lo, hi):
        return co[(co[:, 2] >= z0 + lo * H) & (co[:, 2] <= z0 + hi * H)]

    torso = band(0.55, 0.70)
    torso = torso[np.abs(torso[:, 0] - cx) < 0.12 * H]
    cy = (torso[:, 1].min() + torso[:, 1].max()) / 2 if len(torso) else co[:, 1].mean()
    head = band(0.87, 1.0)
    if len(head):
        chin_y = head[:, 1].max() if fs > 0 else head[:, 1].min()
    else:
        chin_y = cy

    out = {"chin": V((cx, chin_y, z0 + 0.867 * H)), "groin": V((cx, cy, z0 + 0.47 * H))}
    legs = band(0.20, 0.35)
    for side, s in (("L", lx), ("R", -lx)):
        rel = (co[:, 0] - cx) * s
        tip = co[int(np.argmax(rel))]
        shoulder = V((cx + s * 0.115 * H, cy, z0 + 0.82 * H))
        tip_v = V((tip[0], cy, tip[2]))
        d = tip_v - shoulder
        d = d.normalized() if d.length > 1e-6 else V((s, 0, 0))
        wrist = tip_v - d * 0.105 * H
        out["wrist_" + side] = wrist
        out["elbow_" + side] = shoulder + (wrist - shoulder) * 0.5

        leg = legs[((legs[:, 0] - cx) * s > 0.01 * H) & (np.abs(legs[:, 0] - cx) < 0.25 * H)]
        kx = leg[:, 0].mean() if len(leg) else cx + s * 0.095 * H
        ky = (leg[:, 1].min() + leg[:, 1].max()) / 2 if len(leg) else cy
        out["knee_" + side] = V((kx, ky, z0 + 0.28 * H))
    return out


def foot_extent(co, s, cx, z0, H, fs):
    f = co[(co[:, 2] < z0 + 0.07 * H) & ((co[:, 0] - cx) * s > 0.02 * H)]
    if len(f) < 3:
        return None
    if fs < 0:
        return f[:, 1].min(), f[:, 1].max()
    return f[:, 1].max(), f[:, 1].min()  # (toe_y, heel_y)


# --------------------------------------------------------------------------- markers

def get_marker_collection(scene):
    col = bpy.data.collections.get(MARKER_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(MARKER_COLLECTION)
        scene.collection.children.link(col)
    return col


def place_markers(context, mesh, facing):
    co = world_vertices(mesh, context.evaluated_depsgraph_get())
    _, H, _, _ = bounds(co)
    guesses = guess_markers(co, facing)
    col = get_marker_collection(context.scene)
    for key, pos in guesses.items():
        name = MARKER_PREFIX + key
        o = bpy.data.objects.get(name)
        if o is None:
            o = bpy.data.objects.new(name, None)
            o.empty_display_type = "SPHERE"
            o.empty_display_size = 0.015 * H
            o.show_in_front = True
            col.objects.link(o)
        o.location = pos


def read_markers():
    out, missing = {}, []
    bpy.context.view_layer.update()  # matrix_world is stale right after markers are created/moved
    for key in MARKERS:
        o = bpy.data.objects.get(MARKER_PREFIX + key)
        if o is None:
            missing.append(key)
        else:
            out[key] = o.matrix_world.translation.copy()
    return out, missing


def delete_markers():
    for key in MARKERS:
        o = bpy.data.objects.get(MARKER_PREFIX + key)
        if o:
            bpy.data.objects.remove(o, do_unlink=True)
    col = bpy.data.collections.get(MARKER_COLLECTION)
    if col and not col.objects:
        bpy.data.collections.remove(col)


# --------------------------------------------------------------------------- rig builder

FINGERS = [  # name, lateral offset (fraction of H, + = toward front), length factor
    ("Index", 0.015, 0.95), ("Middle", 0.005, 1.0), ("Ring", -0.005, 0.93), ("Pinky", -0.015, 0.75),
]
SEGMENTS = (0.45, 0.30, 0.25)


def build_armature(context, mesh, m, opts):
    """m: marker positions. Returns the armature object."""
    co = world_vertices(mesh, context.evaluated_depsgraph_get())
    z0, H, ztop, cx = bounds(co)
    fs = front_sign(opts["facing"])
    style = opts["naming"]
    V = Vector
    front = V((0.0, fs, 0.0))

    arm_data = bpy.data.armatures.new(mesh.name + "_Armature")
    arm_obj = bpy.data.objects.new(mesh.name + "_Rig", arm_data)
    context.collection.objects.link(arm_obj)
    arm_obj.show_in_front = True
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def add(key, side, head, tail, parent=None, connect=False, roll=FRONT_VEC):
        name = bone_name(style, key, side)
        b = eb.new(name)
        b.head, b.tail = V(head), V(tail)
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + V((0, 0, 1e-3))
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        b.align_roll(roll)
        return name

    groin = m["groin"]
    cy = groin.y
    mid = lambda z: V((cx, cy, z))  # noqa: E731
    pelvis_z = groin.z + 0.04 * H
    spine0_z = pelvis_z + 0.05 * H
    neck_z = m["chin"].z - 0.055 * H
    head_z = m["chin"].z - 0.005 * H

    hips = add("hips", None, mid(pelvis_z), mid(spine0_z))
    prev, ptail = hips, spine0_z
    step = (neck_z - spine0_z) / 3
    for key in ("spine", "spine1", "spine2"):
        prev = add(key, None, mid(ptail), mid(ptail + step), prev, True)
        ptail += step
    spine_top = prev
    neck = add("neck", None, mid(neck_z), mid(head_z), spine_top, True)
    add("head", None, mid(head_z), mid(ztop), neck, True)

    for side in ("L", "R"):
        W, E, K = m["wrist_" + side], m["elbow_" + side], m["knee_" + side]
        # ---- arm
        S = E - (W - E) * 1.1
        clav = add("shoulder", side, V((cx, cy, neck_z - 0.01 * H)), S, spine_top)
        up = add("arm", side, S, E, clav, True)
        fore = add("forearm", side, E, W, up, True)
        d = (W - E).normalized()
        hl = 0.105 * H
        hand = add("hand", side, W, W + d * 0.38 * hl, fore, True)
        if opts["fingers"]:
            base = W + d * 0.38 * hl
            up_roll = V((0, 0, 1))
            for fname, off, lf in FINGERS:
                p = base + front * (off * H)
                parent, pos = hand, p
                for i, seg in enumerate(SEGMENTS, 1):
                    nxt = pos + d * (0.62 * hl * lf * seg)
                    parent = add(f"{fname}{i}", side, pos, nxt, parent, i > 1, up_roll)
                    pos = nxt
            tdir = (d * 0.6 + front * 0.7 + V((0, 0, -0.15))).normalized()
            pos = W + d * 0.10 * hl + front * (0.02 * H)
            parent = hand
            for i, seg in enumerate((0.35, 0.28, 0.22), 1):
                nxt = pos + tdir * (hl * seg)
                parent = add(f"Thumb{i}", side, pos, nxt, parent, i > 1, up_roll)
                pos = nxt

        # ---- leg
        s = 1.0 if K.x > cx else -1.0
        ext = foot_extent(co, s, cx, z0, H, fs)
        toe_y, heel_y = ext if ext else (K.y + fs * 0.12 * H, K.y - fs * 0.05 * H)
        ankle = V((K.x, heel_y + (toe_y - heel_y) * 0.22, z0 + 0.045 * H))
        ball = V((K.x, heel_y + (toe_y - heel_y) * 0.72, z0 + 0.025 * H))
        tip = V((K.x, toe_y, z0 + 0.012 * H))
        hip = V((K.x, cy, pelvis_z))
        upleg = add("upleg", side, hip, K, hips)
        leg = add("leg", side, K, ankle, upleg, True)
        foot = add("foot", side, ankle, ball, leg, True)
        add("toe", side, ball, tip, foot, True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def bind_mesh(context, mesh, arm_obj):
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        return True
    except RuntimeError:
        bpy.ops.object.parent_set(type="ARMATURE_NAME")
        return False


# --------------------------------------------------------------------------- UI / operators

class AutoRigProps(bpy.types.PropertyGroup):
    target: PointerProperty(
        name="Mesh", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
        description="Character mesh to rig (defaults to the active mesh)")
    facing: EnumProperty(
        name="Faces", items=[("NEG_Y", "-Y (Blender front)", ""), ("POS_Y", "+Y", "")],
        default="NEG_Y", description="Direction the character looks. Pose must be T or A")
    naming: EnumProperty(
        name="Bone names",
        items=[("MIXAMO", "Mixamo (mixamorig:)", "Compatible with Mixamo animations / retargeting"),
               ("BLENDER", "Blender (.L / .R)", "Standard Blender side suffixes")],
        default="MIXAMO")
    fingers: BoolProperty(name="Fingers", default=True)
    bind: BoolProperty(name="Bind with automatic weights", default=True)
    keep_markers: BoolProperty(name="Keep markers after rigging", default=False)


def get_target(context):
    p = context.scene.autorig
    if p.target and p.target.type == "MESH":
        return p.target
    ob = context.active_object
    if ob and ob.type == "MESH":
        return ob
    return None


def ensure_object_mode(context):
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


class AUTORIG_OT_place_markers(bpy.types.Operator):
    bl_idname = "autorig.place_markers"
    bl_label = "Place / Reset Markers"
    bl_description = "Estimate joint positions from the mesh and drop adjustable markers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mesh = get_target(context)
        if mesh is None:
            self.report({"ERROR"}, "Select the character mesh first")
            return {"CANCELLED"}
        ensure_object_mode(context)
        place_markers(context, mesh, context.scene.autorig.facing)
        return {"FINISHED"}


class AUTORIG_OT_mirror(bpy.types.Operator):
    bl_idname = "autorig.mirror_markers"
    bl_label = "Mirror Markers"
    bl_options = {"REGISTER", "UNDO"}
    source: EnumProperty(items=[("L", "Left", ""), ("R", "Right", "")], default="L")

    def execute(self, context):
        m, missing = read_markers()
        if missing:
            self.report({"ERROR"}, "Place markers first")
            return {"CANCELLED"}
        cx = m["groin"].x
        dst = "R" if self.source == "L" else "L"
        for kind in ("wrist", "elbow", "knee"):
            p = m[f"{kind}_{self.source}"]
            o = bpy.data.objects[f"{MARKER_PREFIX}{kind}_{dst}"]
            o.location = Vector((2 * cx - p.x, p.y, p.z))
        return {"FINISHED"}


class AUTORIG_OT_clear(bpy.types.Operator):
    bl_idname = "autorig.clear_markers"
    bl_label = "Remove Markers"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        delete_markers()
        return {"FINISHED"}


class AUTORIG_OT_generate(bpy.types.Operator):
    bl_idname = "autorig.generate"
    bl_label = "Generate Rig"
    bl_description = "Build the armature from the markers and bind the mesh"
    bl_options = {"REGISTER", "UNDO"}

    auto_place: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        p = context.scene.autorig
        mesh = get_target(context)
        if mesh is None:
            self.report({"ERROR"}, "Select the character mesh first")
            return {"CANCELLED"}
        ensure_object_mode(context)
        if self.auto_place:
            place_markers(context, mesh, p.facing)
        markers, missing = read_markers()
        if missing:
            self.report({"ERROR"}, "Markers missing - press 'Place / Reset Markers' first")
            return {"CANCELLED"}
        opts = {"facing": p.facing, "naming": p.naming, "fingers": p.fingers}
        arm_obj = build_armature(context, mesh, markers, opts)
        if p.bind:
            if not bind_mesh(context, mesh, arm_obj):
                self.report({"WARNING"}, "Automatic weights failed (mesh may be non-manifold or "
                                         "have loose parts); parented with empty groups instead")
        if not p.keep_markers:
            delete_markers()
        bpy.ops.object.select_all(action="DESELECT")
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
        self.report({"INFO"}, f"Created {len(arm_obj.data.bones)} bones")
        return {"FINISHED"}


class AUTORIG_PT_panel(bpy.types.Panel):
    bl_label = "Auto Rig Biped"
    bl_idname = "AUTORIG_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AutoRig"

    def draw(self, context):
        p = context.scene.autorig
        lay = self.layout
        lay.prop(p, "target")
        lay.prop(p, "facing")
        lay.prop(p, "naming")
        lay.prop(p, "fingers")
        lay.prop(p, "bind")

        box = lay.box()
        op = box.operator("autorig.generate", text="Auto Rig (one click)", icon="ARMATURE_DATA")
        op.auto_place = True

        col = lay.column(align=True)
        col.label(text="Fine-tune (optional):")
        col.operator("autorig.place_markers", icon="EMPTY_AXIS")
        row = col.row(align=True)
        a = row.operator("autorig.mirror_markers", text="Left → Right")
        a.source = "L"
        b = row.operator("autorig.mirror_markers", text="Right → Left")
        b.source = "R"
        col.operator("autorig.generate", text="Generate From Markers").auto_place = False
        col.operator("autorig.clear_markers", icon="X")
        lay.prop(p, "keep_markers")


classes = (AutoRigProps, AUTORIG_OT_place_markers, AUTORIG_OT_mirror, AUTORIG_OT_clear,
           AUTORIG_OT_generate, AUTORIG_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.autorig = PointerProperty(type=AutoRigProps)


def unregister():
    del bpy.types.Scene.autorig
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
