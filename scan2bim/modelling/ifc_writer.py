"""Build the IFC4 model from the parametric component description alone.

    Wall type           IfcWallType with a material layer set; glass walls use IfcCurtainWallType
    Wall                IfcWall (STANDARD or PARTITIONING) or IfcCurtainWall, with 'Axis' and 'Body' representations
    Door, window type   IfcDoorType, IfcWindowType with parametric geometry (lining and leaf, frame and glass)
    Door, window        IfcOpeningElement in the host wall plus IfcDoor or IfcWindow; the type geometry is stretched
                        when the occurrence has its own size; windows in walls with space on one side sit at that face;
                        a door in a gap between elements is placed in the gap
    Floor               IfcSlab FLOOR, extruded downwards by the type thickness
    Ceiling             IfcCovering CEILING at the ceiling height
    Column              IfcColumn with mapped type geometry
    Space               IfcSpace (boundary x height), aggregated into the storey
"""
from __future__ import annotations

import json
from typing import Dict, Sequence

import ifcopenshell
import ifcopenshell.guid
import numpy as np

from .context import ModelContext


class IfcBuilder:
    def __init__(self, description: dict) -> None:
        self.description = description
        self.file = ifcopenshell.file(schema="IFC4")
        self.types: Dict[str, dict] = {}
        self.by_type: Dict[str, list] = {}
        self.contained: list = []
        self.materials: Dict[str, object] = {}
        self.walls: Dict[str, dict] = {}
        self._create_spatial_structure()

    # ---------------------------------------------------------------- small factories
    def point3(self, x: float, y: float, z: float):
        return self.file.createIfcCartesianPoint((float(x), float(y), float(z)))

    def point2(self, x: float, y: float):
        return self.file.createIfcCartesianPoint((float(x), float(y)))

    def direction(self, x: float, y: float, z: float):
        return self.file.createIfcDirection((float(x), float(y), float(z)))

    def axis3(self, origin: Sequence[float], x_axis: Sequence[float] = (1, 0)):
        return self.file.createIfcAxis2Placement3D(self.point3(*origin), self.direction(0, 0, 1), self.direction(x_axis[0], x_axis[1], 0))

    def box(self, cx: float, cy: float, z: float, width: float, depth: float, height: float):
        placement = self.file.createIfcAxis2Placement2D(self.point2(cx, cy), None)
        profile = self.file.createIfcRectangleProfileDef("AREA", None, placement, float(width), float(depth))
        return self.file.createIfcExtrudedAreaSolid(profile, self.axis3((0, 0, z)), self.direction(0, 0, 1), float(height))

    def area_solid(self, outer: Sequence, holes: Sequence, height: float):
        polyline = lambda ring: self.file.createIfcPolyline([self.point2(x, y) for x, y in ring])
        if holes:
            profile = self.file.createIfcArbitraryProfileDefWithVoids("AREA", None, polyline(outer), [polyline(h) for h in holes])
        else:
            profile = self.file.createIfcArbitraryClosedProfileDef("AREA", None, polyline(outer))
        return self.file.createIfcExtrudedAreaSolid(profile, self.axis3((0, 0, 0)), self.direction(0, 0, 1), float(height))

    def body(self, items: list):
        representation = self.file.createIfcShapeRepresentation(self.body_context, "Body", "SweptSolid", items)
        return self.file.createIfcProductDefinitionShape(None, None, [representation])

    def representation_map(self, items: list):
        representation = self.file.createIfcShapeRepresentation(self.body_context, "Body", "SweptSolid", items)
        return self.file.createIfcRepresentationMap(self.axis3((0, 0, 0)), representation)

    def mapped(self, item_map, operator=None):
        operator = operator or self.file.createIfcCartesianTransformationOperator3D(None, None, self.point3(0, 0, 0), None, None)
        item = self.file.createIfcMappedItem(item_map, operator)
        representation = self.file.createIfcShapeRepresentation(self.body_context, "Body", "MappedRepresentation", [item])
        return self.file.createIfcProductDefinitionShape(None, None, [representation])

    def placement(self, relative_to, origin: Sequence[float], x_axis: Sequence[float] = (1, 0)):
        return self.file.createIfcLocalPlacement(relative_to, self.axis3(origin, x_axis))

    def guid(self) -> str:
        return ifcopenshell.guid.new()

    def material(self, name: str):
        if name not in self.materials:
            self.materials[name] = self.file.createIfcMaterial(name, None, None)
        return self.materials[name]

    def properties(self, target, name: str, values: dict) -> None:
        items = []
        for key, value in values.items():
            if isinstance(value, bool):
                nominal = self.file.create_entity("IfcBoolean", value)
            elif isinstance(value, (int, float)):
                nominal = self.file.create_entity("IfcReal", float(value))
            else:
                nominal = self.file.create_entity("IfcLabel", str(value))
            items.append(self.file.createIfcPropertySingleValue(key, None, nominal, None))
        property_set = self.file.createIfcPropertySet(self.guid(), None, name, None, items)
        self.file.createIfcRelDefinesByProperties(self.guid(), None, None, None, [target], property_set)

    def assign(self, occurrence, type_id: str) -> None:
        self.by_type.setdefault(type_id, []).append(occurrence)

    # ---------------------------------------------------------------- project and types
    def _create_spatial_structure(self) -> None:
        f, building, storey = self.file, self.description["Building"], self.description["Storey"]
        units = f.createIfcUnitAssignment([f.createIfcSIUnit(None, unit, None, name) for unit, name in
                                           (("LENGTHUNIT", "METRE"), ("AREAUNIT", "SQUARE_METRE"), ("VOLUMEUNIT", "CUBIC_METRE"), ("PLANEANGLEUNIT", "RADIAN"))])
        context = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, self.axis3((0, 0, 0)), None)
        self.body_context = f.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, context, None, "MODEL_VIEW", None)
        self.axis_context = f.createIfcGeometricRepresentationSubContext("Axis", "Model", None, None, None, None, context, None, "GRAPH_VIEW", None)
        self.project = f.createIfcProject(self.guid(), None, building["name"], "Scan-to-BIM model", None, None, None, [context], units)
        site_placement = self.placement(None, (0, 0, 0))
        site = f.createIfcSite(self.guid(), None, "Site", None, None, site_placement, None, None, "ELEMENT", None, None, None, None, None)
        building_placement = self.placement(site_placement, (0, 0, 0))
        ifc_building = f.createIfcBuilding(self.guid(), None, building["name"], None, None, building_placement, None, None, "ELEMENT", None, None, None)
        self.storey_placement = self.placement(building_placement, (0, 0, storey["elevation"]))
        self.storey = f.createIfcBuildingStorey(self.guid(), None, storey["name"], None, None, self.storey_placement, None, None, "ELEMENT", storey["elevation"])
        for parent, child in ((self.project, site), (site, ifc_building), (ifc_building, self.storey)):
            f.createIfcRelAggregates(self.guid(), None, None, None, parent, [child])

    def door_geometry(self, width: float, height: float) -> list:
        frame = 0.05
        return [self.box(-width / 2 + frame / 2, 0, 0, frame, 0.10, height), self.box(width / 2 - frame / 2, 0, 0, frame, 0.10, height),
                self.box(0, 0, height - frame, width, 0.10, frame), self.box(0, 0, 0, width - 2 * frame, 0.04, height - frame)]

    def window_geometry(self, width: float, height: float) -> list:
        frame = 0.06
        return [self.box(0, 0, 0, width, 0.06, frame), self.box(0, 0, height - frame, width, 0.06, frame),
                self.box(-width / 2 + frame / 2, 0, frame, frame, 0.06, height - 2 * frame),
                self.box(width / 2 - frame / 2, 0, frame, frame, 0.06, height - 2 * frame),
                self.box(0, 0, frame, width - 2 * frame, 0.01, height - 2 * frame)]

    def layer_set(self, material: str, thickness: float, name=None):
        layer = self.file.createIfcMaterialLayer(self.material(material), thickness, None, None, None, None, None)
        return self.file.createIfcMaterialLayerSet([layer], name, None)

    def create_types(self) -> None:
        f, d = self.file, self.description
        for t in d["WallTypes"]:
            thickness = t["thickness"]
            layers = self.layer_set(t["material"], thickness, f"{t['id']} {thickness * 1000:.0f} mm")
            name = f"{t['id']} {t['function']} {thickness * 1000:.0f} mm"
            if t["function"] == "glass_wall":
                entity = f.createIfcCurtainWallType(self.guid(), None, name, t["evidence"], None, None, None, None, None, "NOTDEFINED")
            else:
                predefined = "PARTITIONING" if t["function"] == "partition" else "STANDARD"
                entity = f.createIfcWallType(self.guid(), None, name, t["evidence"], None, None, None, None, None, predefined)
            f.createIfcRelAssociatesMaterial(self.guid(), None, None, None, [entity], layers)
            self.types[t["id"]] = dict(entity=entity, thickness=thickness, function=t["function"], layers=layers)
        for t in d["DoorTypes"]:
            item_map = self.representation_map(self.door_geometry(t["width"], t["height"]))
            name = f"{t['id']} door {t['width'] * 1000:.0f}x{t['height'] * 1000:.0f}"
            entity = f.createIfcDoorType(self.guid(), None, name, None, None, None, [item_map], None, None, "DOOR", t["operation"], False, None)
            self.types[t["id"]] = dict(entity=entity, width=t["width"], height=t["height"], map=item_map)
        for t in d["WindowTypes"]:
            item_map = self.representation_map(self.window_geometry(t["width"], t["height"]))
            name = f"{t['id']} window {t['width'] * 1000:.0f}x{t['height'] * 1000:.0f}"
            entity = f.createIfcWindowType(self.guid(), None, name, None, None, None, [item_map], None, None, "WINDOW", t["operation"], False, None)
            self.types[t["id"]] = dict(entity=entity, width=t["width"], height=t["height"], map=item_map)
        for t in d["FloorTypes"]:
            entity = f.createIfcSlabType(self.guid(), None, f"{t['id']} floor {t['thickness'] * 1000:.0f} mm", None, None, None, None, None, None, "FLOOR")
            f.createIfcRelAssociatesMaterial(self.guid(), None, None, None, [entity], self.layer_set(t["material"], t["thickness"]))
            self.types[t["id"]] = dict(entity=entity, thickness=t["thickness"])
        for t in d["CeilingTypes"]:
            entity = f.createIfcCoveringType(self.guid(), None, f"{t['id']} ceiling {t['thickness'] * 1000:.0f} mm", None, None, None, None, None, None, "CEILING")
            f.createIfcRelAssociatesMaterial(self.guid(), None, None, None, [entity], self.material(t["material"]))
            self.types[t["id"]] = dict(entity=entity, thickness=t["thickness"])
        height = d["Storey"]["height"]
        for t in d["ColumnTypes"]:
            item_map = self.representation_map([self.box(0, 0, 0, t["width"], t["depth"], height)])
            name = f"{t['id']} column {t['width'] * 1000:.0f}x{t['depth'] * 1000:.0f}"
            entity = f.createIfcColumnType(self.guid(), None, name, None, None, None, [item_map], None, None, "COLUMN")
            self.types[t["id"]] = dict(entity=entity, map=item_map)
        f.createIfcRelDeclares(self.guid(), None, None, None, self.project, [t["entity"] for t in self.types.values()])

    # ---------------------------------------------------------------- occurrences
    def create_walls(self) -> None:
        f = self.file
        for wall in self.description["Walls"]:
            kind = self.types[wall["typeID"]]
            start, end = np.array(wall["start"]), np.array(wall["end"])
            length = float(np.linalg.norm(end - start))
            direction = (end - start) / length
            placement = self.placement(self.storey_placement, (start[0], start[1], wall["baseOffset"]), direction)
            axis = f.createIfcShapeRepresentation(self.axis_context, "Axis", "Curve2D", [f.createIfcPolyline([self.point2(0, 0), self.point2(length, 0)])])
            body = f.createIfcShapeRepresentation(self.body_context, "Body", "SweptSolid", [self.box(length / 2, 0, 0, length, kind["thickness"], wall["height"])])
            shape = f.createIfcProductDefinitionShape(None, None, [axis, body])
            external = kind["function"] == "exterior" or (kind["function"] == "glass_wall" and wall["spaceSide"] != "both")
            if kind["function"] == "glass_wall":
                entity = f.createIfcCurtainWall(self.guid(), None, wall["name"], None, None, placement, shape, wall["id"], "NOTDEFINED")
                self.properties(entity, "Pset_CurtainWallCommon", {"Reference": wall["typeID"], "IsExternal": external})
            else:
                predefined = "PARTITIONING" if kind["function"] == "partition" else "STANDARD"
                entity = f.createIfcWall(self.guid(), None, wall["name"], None, None, placement, shape, wall["id"], predefined)
                usage = f.createIfcMaterialLayerSetUsage(kind["layers"], "AXIS2", "POSITIVE", -kind["thickness"] / 2, None)
                f.createIfcRelAssociatesMaterial(self.guid(), None, None, None, [entity], usage)
                self.properties(entity, "Pset_WallCommon", {"Reference": wall["typeID"], "IsExternal": external})
            self.properties(entity, "Scan_Provenance", {"Source": wall["source"], "Length_m": round(length, 3), "Height_m": wall["height"]})
            self.assign(entity, wall["typeID"])
            self.contained.append(entity)
            self.walls[wall["id"]] = dict(entity=entity, placement=placement, thickness=kind["thickness"], side=wall["spaceSide"], base=wall["baseOffset"])

    def create_opening(self, item: dict, ifc_class: str) -> None:
        f = self.file
        kind = self.types[item["typeID"]]
        width, height, sill = item["width"], item["height"], item["sill"]
        wall = self.walls.get(item["wallID"])
        opening = None
        if wall is None:
            angle = np.radians(item["rotation"])
            filling_placement = self.placement(self.storey_placement, (item["location"][0], item["location"][1], sill), (np.cos(angle), np.sin(angle)))
        else:
            placement = self.placement(wall["placement"], (item["offset"], 0, sill - wall["base"]))
            void = self.body([self.box(0, 0, 0, width, wall["thickness"] + 0.1, height)])
            opening = f.createIfcOpeningElement(self.guid(), None, f"Opening {item['name']}", None, None, placement, void, None, "OPENING")
            f.createIfcRelVoidsElement(self.guid(), None, None, None, wall["entity"], opening)
            shift = 0.0
            if ifc_class == "IfcWindow" and wall["side"] in ("left", "right"):      # glass at the face of the space side
                shift = (wall["thickness"] / 2 - 0.03) * (1 if wall["side"] == "left" else -1)
            filling_placement = self.placement(placement, (0, shift, 0))
        operator = None
        if abs(width - kind["width"]) > 1e-3 or abs(height - kind["height"]) > 1e-3:
            operator = f.createIfcCartesianTransformationOperator3DnonUniform(None, None, self.point3(0, 0, 0), width / kind["width"], None, 1.0, height / kind["height"])
        entity = f.create_entity(ifc_class, GlobalId=self.guid(), Name=item["name"], Tag=item["id"], ObjectPlacement=filling_placement,
                                 Representation=self.mapped(kind["map"], operator), OverallHeight=height, OverallWidth=width,
                                 PredefinedType="DOOR" if ifc_class == "IfcDoor" else "WINDOW")
        common = "Pset_DoorCommon" if ifc_class == "IfcDoor" else "Pset_WindowCommon"
        values = {"Reference": item["typeID"]}
        if opening is not None:
            f.createIfcRelFillsElement(self.guid(), None, None, None, opening, entity)
            values["IsExternal"] = wall["side"] in ("left", "right")
        self.properties(entity, common, values)
        self.properties(entity, "Scan_Provenance", {"Source": item["source"], "Width_m": width, "Height_m": height, "Sill_m": sill})
        self.assign(entity, item["typeID"])
        self.contained.append(entity)

    def create_surfaces_and_columns(self) -> None:
        f, d = self.file, self.description
        for item in d["Floors"]:
            kind = self.types[item["typeID"]]
            shape = self.body([self.area_solid(item["boundary"], item["holes"], kind["thickness"])])
            placement = self.placement(self.storey_placement, (0, 0, -kind["thickness"]))
            entity = f.createIfcSlab(self.guid(), None, item["name"], None, None, placement, shape, item["id"], "FLOOR")
            self.assign(entity, item["typeID"])
            self.contained.append(entity)
        for item in d["Ceilings"]:
            kind = self.types[item["typeID"]]
            shape = self.body([self.area_solid(item["boundary"], item["holes"], kind["thickness"])])
            placement = self.placement(self.storey_placement, (0, 0, item["height"]))
            entity = f.createIfcCovering(self.guid(), None, item["name"], None, None, placement, shape, item["id"], "CEILING")
            self.properties(entity, "Scan_Provenance", {"Source": item["source"], "Height_m": item["height"]})
            self.assign(entity, item["typeID"])
            self.contained.append(entity)
        for item in d["Columns"]:
            angle = np.radians(item["rotation"])
            placement = self.placement(self.storey_placement, (item["location"][0], item["location"][1], 0), (np.cos(angle), np.sin(angle)))
            entity = f.createIfcColumn(self.guid(), None, item["name"], None, None, placement, self.mapped(self.types[item["typeID"]]["map"]), item["id"], "COLUMN")
            self.properties(entity, "Scan_Provenance", {"Source": item["source"], "Height_m": item["height"]})
            self.assign(entity, item["typeID"])
            self.contained.append(entity)

    def create_spaces(self) -> None:
        f = self.file
        spaces = []
        for item in self.description["Spaces"]:
            shape = self.body([self.area_solid(item["boundary"], item["holes"], item["height"])])
            placement = self.placement(self.storey_placement, (0, 0, 0))
            entity = f.createIfcSpace(self.guid(), None, item["name"], None, None, placement, shape, None, "ELEMENT", "INTERNAL", 0.0)
            self.properties(entity, "Scan_Provenance", {"Source": "scanner visibility and graph cut", "Area_m2": item["area"], "Height_m": item["height"]})
            spaces.append(entity)
        if spaces:
            f.createIfcRelAggregates(self.guid(), None, None, None, self.storey, spaces)

    def build(self):
        self.create_types()
        self.create_walls()
        for item in self.description["Doors"]:
            self.create_opening(item, "IfcDoor")
        for item in self.description["Windows"]:
            self.create_opening(item, "IfcWindow")
        self.create_surfaces_and_columns()
        self.create_spaces()
        for type_id, occurrences in self.by_type.items():
            self.file.createIfcRelDefinesByType(self.guid(), None, None, None, occurrences, self.types[type_id]["entity"])
        self.file.createIfcRelContainedInSpatialStructure(self.guid(), None, None, None, self.contained, self.storey)
        return self.file


def run(ctx: ModelContext) -> None:
    with open(ctx.output / "components.json", encoding="utf-8") as handle:
        description = json.load(handle)
    model = IfcBuilder(description).build()
    path = ctx.output / "model.ifc"
    model.write(str(path))
    classes = ["IfcSlab", "IfcCovering", "IfcWall", "IfcCurtainWall", "IfcDoor", "IfcWindow", "IfcColumn", "IfcSpace", "IfcOpeningElement"]
    ctx.log(f"written {path}: " + ", ".join(f"{name} {len(model.by_type(name))}" for name in classes))
