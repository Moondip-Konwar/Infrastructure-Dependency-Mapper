#!/usr/bin/env python3
"""Generate mock OpenStreetMap data in the Overpass API JSON format.

Produces a synthetic road network with the same JSON layout as an Overpass
response (`out body; >; out skel qt;`): a top-level object with "version",
"generator", "osm3s" and "elements", where ways come first and the nodes they
reference come afterwards. Readers must index nodes by id before resolving ways.

Generated area:
* A grid of multi-node road ways (primary/secondary/tertiary/residential)
  that share nodes at intersections.
* A river (waterway=river) running east-west through the middle. Roads cannot
  cross it except over a few bridges (bridge=yes), so bridges are single points
  of failure.
* Localities (place=village), buildings clustered around them, hospitals
  (amenity=hospital, closed ways) and clinics (amenity=clinic, nodes).

Examples:
    python scripts/generate_mock_osm.py
    python scripts/generate_mock_osm.py --bridge-count 2 --building-count 1500
    python scripts/generate_mock_osm.py --config my_settings.json

Settings are applied in this order, later winning: defaults, --config JSON file,
command line options. Requires Python 3.10+ and no third-party packages.
"""

import argparse
import dataclasses
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src" / "lib" / "data" / "osm_data.json"

METERS_PER_DEGREE_LATITUDE = 111_320.0
COORDINATE_DECIMAL_PLACES = 7
MAX_PLACEMENT_ATTEMPTS = 50

# Residential roads intentionally have no maxspeed, as is common in real data.
MAXSPEED_KMH_BY_HIGHWAY = {"primary": "60", "secondary": "50", "tertiary": "40"}
BUILDING_TYPES, BUILDING_TYPE_WEIGHTS = zip(
    ("house", 55), ("yes", 25), ("residential", 8), ("apartments", 5), ("commercial", 4), ("retail", 3)
)


def option(default: Any, help_text: str, low: float | None = None, high: float | None = None) -> Any:
    """Declare a setting with its help text and optional inclusive bounds."""
    return field(default=default, metadata={"help": help_text, "low": low, "high": high})


@dataclass(frozen=True)
class GeneratorConfig:
    """Every tunable setting. Field names are the CLI options (dashes) and JSON keys."""

    seed: int = option(42, "Random seed. Same seed and settings give identical output.")

    # Bounding box. The default covers the Guwahati area in Kamrup, Assam.
    min_lat: float = option(26.10, "Southern edge of the bounding box, in degrees.", -90, 90)
    max_lat: float = option(26.22, "Northern edge of the bounding box, in degrees.", -90, 90)
    min_lon: float = option(91.62, "Western edge of the bounding box, in degrees.", -180, 180)
    max_lon: float = option(91.82, "Eastern edge of the bounding box, in degrees.", -180, 180)

    # Road network.
    grid_rows: int = option(12, "Number of east-west roads. The river runs between the middle two.", 4)
    grid_columns: int = option(16, "Number of north-south roads.", 2)
    margin_fraction: float = option(0.08, "Empty border inside the box, as a fraction of its size.", 0, 0.25)
    position_jitter_fraction: float = option(
        0.15, "How far intersections wander from the perfect grid, as a fraction of grid spacing.", 0, 0.3
    )
    shape_nodes_per_segment: int = option(
        2, "Extra nodes between two intersections. Raise it for longer ways with more nodes.", 0
    )
    shape_node_offset_meters: float = option(4.0, "Sideways wobble of those extra nodes, in meters.", 0)
    road_dropout_probability: float = option(0.05, "Chance that a block-length road segment is missing.", 0, 0.5)
    arterial_interval: int = option(4, "Every Nth east-west and north-south road becomes a secondary road.", 1)

    # River and bridges.
    river_name: str = option("Mock River", "Name tag of the river.")
    river_node_count: int = option(120, "Number of nodes in the river way.", 2)
    river_meander_fraction: float = option(0.2, "River wiggle amplitude, as a fraction of the gap between road rows.", 0)
    bridge_count: int = option(3, "Number of bridges over the river (at most grid_columns).", 1)

    # Buildings and localities.
    locality_count: int = option(6, "Number of localities (place=village nodes).", 0)
    building_count: int = option(800, "Number of ordinary buildings to place.", 0)
    locality_spread_meters: float = option(
        700.0, "Standard deviation of building distance from its locality center, in meters.", 1
    )
    scatter_fraction: float = option(0.15, "Fraction of buildings placed anywhere instead of near a locality.", 0, 1)
    building_size_min_meters: float = option(7.0, "Smallest building side length, in meters.", 0.1)
    building_size_max_meters: float = option(16.0, "Largest building side length, in meters.", 0.1)
    river_exclusion_meters: float = option(150.0, "Buildings and localities stay this far from the river.", 0)

    # Health facilities.
    hospital_count: int = option(3, "Number of hospitals (building footprints).", 0)
    clinic_count: int = option(4, "Number of clinics (single nodes).", 0)

    # Identifier ranges. Real OpenStreetMap ids are positive integers.
    first_node_id: int = option(100_000_000, "Id of the first generated node.", 1)
    first_way_id: int = option(200_000_000, "Id of the first generated way.", 1)

    def __post_init__(self) -> None:
        errors = []
        for item in dataclasses.fields(self):
            value, low, high = getattr(self, item.name), item.metadata.get("low"), item.metadata.get("high")
            if (low is not None and value < low) or (high is not None and value > high):
                errors.append(f"{item.name}={value} is invalid, it must be {describe_bounds(low, high)}")
        checks = [
            (self.min_lat < self.max_lat, "min_lat must be below max_lat"),
            (self.min_lon < self.max_lon, "min_lon must be below max_lon"),
            (self.building_size_min_meters <= self.building_size_max_meters, "building size min must not exceed max"),
            (self.bridge_count <= self.grid_columns, "bridge_count must not exceed grid_columns"),
            (
                self.river_meander_fraction + self.position_jitter_fraction <= 0.4,
                "river_meander_fraction + position_jitter_fraction must be at most 0.4, "
                "otherwise roads could end up on the wrong side of the river",
            ),
        ]
        errors += [message for is_ok, message in checks if not is_ok]
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def describe_bounds(low: float | None, high: float | None) -> str:
    if low is None:
        return f"at most {high}"
    return f"at least {low}" if high is None else f"between {low} and {high}"


def road_tags(highway: str, name: str) -> dict[str, str]:
    tags = {"highway": highway, "name": name}
    if highway in MAXSPEED_KMH_BY_HIGHWAY:
        tags["maxspeed"] = MAXSPEED_KMH_BY_HIGHWAY[highway]
    return tags


def offset_position(lat: float, lon: float, east_meters: float, north_meters: float) -> tuple[float, float]:
    return (
        lat + north_meters / METERS_PER_DEGREE_LATITUDE,
        lon + east_meters / (METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(lat))),
    )


class MockOsmBuilder:
    """Builds nodes and ways, then serializes them to the Overpass JSON layout."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.coordinates: list[tuple[float, float]] = []  # index = node id - first_node_id
        self.node_tags: dict[int, dict[str, str]] = {}
        self.ways: list[dict[str, Any]] = []
        self.grid: list[list[int]] = []  # grid[row][column] is an intersection node id
        self._compute_layout()
        self.bridge_columns = sorted(self.rng.sample(range(config.grid_columns), config.bridge_count))
        self._river_phases = (self.rng.uniform(0, 2 * math.pi), self.rng.uniform(0, 2 * math.pi))

    def _compute_layout(self) -> None:
        c = self.config
        lat_span, lon_span = c.max_lat - c.min_lat, c.max_lon - c.min_lon
        self.first_lat = c.min_lat + lat_span * c.margin_fraction
        self.first_lon = c.min_lon + lon_span * c.margin_fraction
        grid_lat_span = lat_span * (1 - 2 * c.margin_fraction)
        grid_lon_span = lon_span * (1 - 2 * c.margin_fraction)
        self.last_lat, self.last_lon = self.first_lat + grid_lat_span, self.first_lon + grid_lon_span
        self.row_step_lat = grid_lat_span / (c.grid_rows - 1)
        self.column_step_lon = grid_lon_span / (c.grid_columns - 1)

        row_step_meters = self.row_step_lat * METERS_PER_DEGREE_LATITUDE
        column_step_meters = (
            self.column_step_lon * METERS_PER_DEGREE_LATITUDE * math.cos(math.radians((c.min_lat + c.max_lat) / 2))
        )
        self.jitter_meters = c.position_jitter_fraction * min(row_step_meters, column_step_meters)

        # The river lies in the gap between the last south row and the first north row.
        self.first_north_row = c.grid_rows // 2
        self.river_center_lat = self.first_lat + self.row_step_lat * (self.first_north_row - 0.5)
        self.river_amplitude_lat = c.river_meander_fraction * self.row_step_lat

        worst_case_clearance = row_step_meters * (0.5 - c.river_meander_fraction - c.position_jitter_fraction)
        if worst_case_clearance <= c.shape_node_offset_meters:
            raise ValueError(
                "Grid rows are too close together for this river and jitter. Increase the bounding box "
                "height, reduce grid_rows, or reduce river_meander_fraction, position_jitter_fraction "
                "or shape_node_offset_meters."
            )

    def river_latitude_at(self, longitude: float) -> float:
        progress = (longitude - self.config.min_lon) / (self.config.max_lon - self.config.min_lon)
        wave = 0.7 * math.sin(2 * math.pi * 1.5 * progress + self._river_phases[0])
        wave += 0.3 * math.sin(2 * math.pi * 3.5 * progress + self._river_phases[1])
        return self.river_center_lat + wave * self.river_amplitude_lat

    # --------------------------------------------------------------- primitives

    def _add_node(self, lat: float, lon: float, tags: dict[str, str] | None = None) -> int:
        node_id = self.config.first_node_id + len(self.coordinates)
        self.coordinates.append((lat, lon))
        if tags:
            self.node_tags[node_id] = tags
        return node_id

    def _add_way(self, node_ids: list[int], tags: dict[str, str]) -> None:
        way_id = self.config.first_way_id + len(self.ways)
        self.ways.append({"type": "way", "id": way_id, "nodes": node_ids, "tags": tags})

    def _position(self, node_id: int) -> tuple[float, float]:
        return self.coordinates[node_id - self.config.first_node_id]

    def _uniform_point(self) -> tuple[float, float]:
        return (
            self.rng.uniform(self.first_lat, self.last_lat),
            self.rng.uniform(self.first_lon, self.last_lon),
        )

    def _sample_site(self, draw_point: Callable[[], tuple[float, float]]) -> tuple[float, float] | None:
        """Draw points until one lies inside the grid area and away from the river."""
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            lat, lon = draw_point()
            inside = self.first_lat <= lat <= self.last_lat and self.first_lon <= lon <= self.last_lon
            river_distance = abs(lat - self.river_latitude_at(lon)) * METERS_PER_DEGREE_LATITUDE
            if inside and river_distance >= self.config.river_exclusion_meters:
                return lat, lon
        return None

    # ------------------------------------------------------------------- roads

    def _create_shape_nodes(self, start_id: int, end_id: int) -> list[int]:
        """Create wobbly nodes along the straight line between two intersections."""
        (start_lat, start_lon), (end_lat, end_lon) = self._position(start_id), self._position(end_id)
        east = (end_lon - start_lon) * METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(start_lat))
        north = (end_lat - start_lat) * METERS_PER_DEGREE_LATITUDE
        length = math.hypot(east, north)
        count, offset = self.config.shape_nodes_per_segment, self.config.shape_node_offset_meters

        node_ids = []
        for index in range(1, count + 1):
            progress = index / (count + 1)
            sideways = self.rng.uniform(-offset, offset)
            lat, lon = offset_position(
                start_lat + (end_lat - start_lat) * progress,
                start_lon + (end_lon - start_lon) * progress,
                -north / length * sideways,
                east / length * sideways,
            )
            node_ids.append(self._add_node(lat, lon))
        return node_ids

    def _create_edge(self, start_id: int, end_id: int) -> list[int] | None:
        """Shape node ids between two intersections, or None if this road piece is missing."""
        if self.rng.random() < self.config.road_dropout_probability:
            return None
        return self._create_shape_nodes(start_id, end_id)

    def _add_road_chain(self, chain: list[int], edges: list[list[int] | None], tags: dict[str, str]) -> None:
        """Add ways along a chain of intersections, cutting wherever a piece is missing."""
        current = [chain[0]]
        for next_intersection, shape_node_ids in zip(chain[1:], edges):
            if shape_node_ids is None:
                if len(current) > 1:
                    self._add_way(current, tags)
                current = [next_intersection]
            else:
                current += [*shape_node_ids, next_intersection]
        if len(current) > 1:
            self._add_way(current, tags)

    def _classify_road(self, index: int, is_bridge_road: bool = False) -> str:
        if is_bridge_road:
            return "primary"
        if index % self.config.arterial_interval == 0:
            return "secondary"
        return "tertiary" if index % 2 == 0 else "residential"

    def _create_roads(self) -> None:
        c = self.config
        for row in range(c.grid_rows):
            self.grid.append(
                [
                    self._add_node(
                        *offset_position(
                            self.first_lat + row * self.row_step_lat,
                            self.first_lon + column * self.column_step_lon,
                            self.rng.uniform(-self.jitter_meters, self.jitter_meters),
                            self.rng.uniform(-self.jitter_meters, self.jitter_meters),
                        )
                    )
                    for column in range(c.grid_columns)
                ]
            )

        horizontal = {
            (row, column): self._create_edge(self.grid[row][column], self.grid[row][column + 1])
            for row in range(c.grid_rows)
            for column in range(c.grid_columns - 1)
        }
        # The pieces that would cross the river are replaced by bridges (or nothing).
        vertical = {
            (row, column): self._create_edge(self.grid[row][column], self.grid[row + 1][column])
            for row in range(c.grid_rows - 1)
            if row != self.first_north_row - 1
            for column in range(c.grid_columns)
        }

        for row in range(c.grid_rows):
            self._add_road_chain(
                self.grid[row],
                [horizontal[(row, column)] for column in range(c.grid_columns - 1)],
                road_tags(self._classify_road(row), f"East-West Road {row + 1}"),
            )
        # Each north-south road is cut in two by the river: a south part and a north part.
        parts = (range(self.first_north_row), range(self.first_north_row, c.grid_rows))
        for column in range(c.grid_columns):
            tags = road_tags(
                self._classify_road(column, column in self.bridge_columns), f"North-South Road {column + 1}"
            )
            for rows in parts:
                self._add_road_chain(
                    [self.grid[row][column] for row in rows],
                    [vertical[(row, column)] for row in rows[:-1]],
                    tags,
                )

    def _create_bridges(self) -> None:
        middle_node_count = max(1, self.config.shape_nodes_per_segment)
        for bridge_number, column in enumerate(self.bridge_columns, start=1):
            south_id = self.grid[self.first_north_row - 1][column]
            north_id = self.grid[self.first_north_row][column]
            (south_lat, south_lon), (north_lat, north_lon) = self._position(south_id), self._position(north_id)
            middle_ids = []
            for index in range(1, middle_node_count + 1):
                progress = index / (middle_node_count + 1)
                middle_ids.append(
                    self._add_node(
                        south_lat + (north_lat - south_lat) * progress,
                        south_lon + (north_lon - south_lon) * progress,
                    )
                )
            tags = road_tags("primary", f"Bridge {bridge_number}") | {"bridge": "yes", "layer": "1"}
            self._add_way([south_id, *middle_ids, north_id], tags)

    def _create_river(self) -> None:
        c = self.config
        longitudes = [
            c.min_lon + (c.max_lon - c.min_lon) * index / (c.river_node_count - 1)
            for index in range(c.river_node_count)
        ]
        node_ids = [self._add_node(self.river_latitude_at(lon), lon) for lon in longitudes]
        self._add_way(node_ids, {"waterway": "river", "name": c.river_name})

    # --------------------------------------------------- localities and buildings

    def _add_footprint(
        self,
        center_lat: float,
        center_lon: float,
        width_meters: float,
        depth_meters: float,
        rotation_radians: float,
        tags: dict[str, str],
    ) -> None:
        """Add a closed rectangular way: four corner nodes, first node repeated at the end."""
        half_width, half_depth = width_meters / 2, depth_meters / 2
        cosine, sine = math.cos(rotation_radians), math.sin(rotation_radians)
        corner_ids = [
            self._add_node(
                *offset_position(
                    center_lat,
                    center_lon,
                    corner_east * cosine - corner_north * sine,
                    corner_east * sine + corner_north * cosine,
                )
            )
            for corner_east, corner_north in (
                (-half_width, -half_depth),
                (half_width, -half_depth),
                (half_width, half_depth),
                (-half_width, half_depth),
            )
        ]
        self._add_way([*corner_ids, corner_ids[0]], tags)

    def _random_rotation(self) -> float:
        """Mostly axis-aligned, like buildings along a street grid, with some tilt."""
        return self.rng.choice([0.0, math.pi / 2]) + self.rng.uniform(-0.35, 0.35)

    def _create_localities(self) -> list[tuple[float, float, float]]:
        """Create place nodes. Returns (lat, lon, weight) for each one actually placed."""
        localities = []
        for number in range(1, self.config.locality_count + 1):
            site = self._sample_site(self._uniform_point)
            if site is None:
                continue
            self._add_node(*site, {"place": "village", "name": f"Locality {number}"})
            localities.append((*site, self.rng.uniform(0.5, 1.5)))
        return localities

    def _create_buildings(self, localities: list[tuple[float, float, float]]) -> None:
        c = self.config
        weights = [weight for _, _, weight in localities]

        def draw_point() -> tuple[float, float]:
            if not localities or self.rng.random() < c.scatter_fraction:
                return self._uniform_point()
            lat, lon, _ = self.rng.choices(localities, weights=weights)[0]
            spread = c.locality_spread_meters
            return offset_position(lat, lon, self.rng.gauss(0, spread), self.rng.gauss(0, spread))

        for _ in range(c.building_count):
            site = self._sample_site(draw_point)
            if site is not None:
                self._add_footprint(
                    *site,
                    self.rng.uniform(c.building_size_min_meters, c.building_size_max_meters),
                    self.rng.uniform(c.building_size_min_meters, c.building_size_max_meters),
                    self._random_rotation(),
                    {"building": self.rng.choices(BUILDING_TYPES, weights=BUILDING_TYPE_WEIGHTS)[0]},
                )

    # -------------------------------------------------------- health facilities

    def _block_center(self, row: int, column: int) -> tuple[float, float]:
        corners = [
            self._position(self.grid[corner_row][corner_column])
            for corner_row in (row, row + 1)
            for corner_column in (column, column + 1)
        ]
        return sum(lat for lat, _ in corners) / 4, sum(lon for _, lon in corners) / 4

    def _create_health_facilities(self) -> None:
        c = self.config
        blocks_per_row = range(c.grid_columns - 1)
        south_blocks = [(row, column) for row in range(self.first_north_row - 1) for column in blocks_per_row]
        north_blocks = [(row, column) for row in range(self.first_north_row, c.grid_rows - 1) for column in blocks_per_row]
        if c.hospital_count + c.clinic_count > len(south_blocks) + len(north_blocks):
            raise ValueError("Not enough road blocks for the requested hospitals and clinics.")
        self.rng.shuffle(south_blocks)
        self.rng.shuffle(north_blocks)

        # Hospitals alternate between the two river banks, so both sides are served.
        for number in range(1, c.hospital_count + 1):
            preferred, other = (south_blocks, north_blocks) if number % 2 == 1 else (north_blocks, south_blocks)
            self._add_footprint(
                *self._block_center(*(preferred or other).pop()),
                self.rng.uniform(30, 50),
                self.rng.uniform(25, 40),
                self._random_rotation(),
                {
                    "amenity": "hospital",
                    "healthcare": "hospital",
                    "building": "hospital",
                    "emergency": "yes",
                    "name": f"Mock Hospital {number}",
                },
            )

        remaining_blocks = south_blocks + north_blocks
        self.rng.shuffle(remaining_blocks)
        for number in range(1, c.clinic_count + 1):
            self._add_node(
                *self._block_center(*remaining_blocks.pop()),
                {"amenity": "clinic", "healthcare": "clinic", "name": f"Mock Clinic {number}"},
            )

    # ------------------------------------------------------------------ output

    def build(self) -> dict[str, Any]:
        self._create_roads()
        self._create_bridges()
        self._create_river()
        self._create_buildings(self._create_localities())
        self._create_health_facilities()

        # Intersections whose every road piece was dropped belong to no way. Real Overpass
        # output never contains such nodes unless they carry tags.
        referenced_ids = {node_id for way in self.ways for node_id in way["nodes"]}
        node_elements = []
        for index, (lat, lon) in enumerate(self.coordinates):
            node_id = self.config.first_node_id + index
            tags = self.node_tags.get(node_id)
            if node_id in referenced_ids or tags:
                element = {
                    "type": "node",
                    "id": node_id,
                    "lat": round(lat, COORDINATE_DECIMAL_PLACES),
                    "lon": round(lon, COORDINATE_DECIMAL_PLACES),
                }
                node_elements.append(element | {"tags": tags} if tags else element)

        return {
            "version": 0.6,
            "generator": "generate_mock_osm.py (mock data, not fetched from the Overpass API)",
            "osm3s": {
                "timestamp_osm_base": "2026-01-01T00:00:00Z",
                "copyright": "Mock data. Not derived from OpenStreetMap.",
            },
            "elements": self.ways + node_elements,
        }


def write_document(document: dict[str, Any], output_path: Path) -> None:
    """Write the document with one element per line, which keeps git diffs readable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(
        f"  {json.dumps(key)}: {json.dumps(value)},\n" for key, value in document.items() if key != "elements"
    )
    body = ",\n".join("    " + json.dumps(element, separators=(",", ":")) for element in document["elements"])
    output_path.write_text("{\n" + header + '  "elements": [\n' + body + "\n  ]\n}\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate mock OpenStreetMap data in the Overpass API JSON format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to write the JSON file.")
    parser.add_argument(
        "--config", type=Path, help="JSON file with settings. Keys are the option names with underscores."
    )
    # argparse.SUPPRESS keeps unset options out of the result, so command line values
    # override the config file but untouched defaults never do.
    for item in dataclasses.fields(GeneratorConfig):
        low, high = item.metadata["low"], item.metadata["high"]
        bounds = "" if low is None and high is None else f", {describe_bounds(low, high)}"
        parser.add_argument(
            "--" + item.name.replace("_", "-"),
            dest=item.name,
            type=item.type,
            default=argparse.SUPPRESS,
            help=f"{item.metadata['help']} (default: {item.default}{bounds})",
        )
    return parser


def load_config_file(config_path: Path) -> dict[str, Any]:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    unknown_names = set(values) - {item.name for item in dataclasses.fields(GeneratorConfig)}
    if unknown_names:
        raise ValueError(f"Unknown setting(s) in {config_path}: {', '.join(sorted(unknown_names))}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = vars(parser.parse_args(argv))
    output_path: Path = arguments.pop("output")
    config_path: Path | None = arguments.pop("config")

    try:
        settings = load_config_file(config_path) if config_path else {}
        config = GeneratorConfig(**(settings | arguments))
        builder = MockOsmBuilder(config)
        document = builder.build()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    write_document(document, output_path)

    counts = Counter()
    for element in document["elements"]:
        tags = element.get("tags", {})
        counts[element["type"]] += 1
        counts["bridge"] += tags.get("bridge") == "yes"
        counts["road"] += "highway" in tags and "bridge" not in tags
        counts["hospital"] += tags.get("amenity") == "hospital"
        counts["clinic"] += tags.get("amenity") == "clinic"
        counts["locality"] += tags.get("place") == "village"
        counts["building"] += "building" in tags and tags.get("amenity") != "hospital"
    print(f"Wrote {output_path}")
    print(f"  elements:   {len(document['elements'])} ({counts['way']} ways, {counts['node']} nodes)")
    print(f"  roads:      {counts['road']} ways, {counts['bridge']} bridges")
    print(f"  bridges at: north-south road numbers {[column + 1 for column in builder.bridge_columns]}")
    print(f"  buildings:  {counts['building']} of {config.building_count} placed")
    print(f"  places:     {counts['locality']} localities, {counts['hospital']} hospitals, {counts['clinic']} clinics")
    return 0


if __name__ == "__main__":
    sys.exit(main())