"""Generate matched, seeded Catanatron games and snapshot openings before dice."""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from catanatron.game import Game
from catanatron.models.enums import RESOURCES, ActionType
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer


RULES = ("basic", "advanced")
SIZES = (2, 4)
STRATEGIES = {
    "production": "total_pips",
    "diversity": "resource_diversity",
    "road_materials": "road_material_pips",
    "city_materials": "city_material_pips",
    "expansion": "best_expansion_pips",
    "port_fit": "port_fit_pips",
    "balanced": "balanced_score",
}
FEATURES = (
    "total_pips", "first_pips", "second_pips", "resource_diversity",
    "unique_roll_numbers", "starting_cards", "starting_road_cards",
    "generic_port", "port_fit_pips", "expansion_sites", "best_expansion_pips",
    "road_material_pips", "city_material_pips", "balanced_score",
    *(f"pips_{resource.lower()}" for resource in RESOURCES),
)


def pips(tile):
    return 6 - abs(7 - tile.number) if tile.number else 0


def setup_game(game, seed, player_count):
    """Use the same opening decisions for both simulated skill tiers."""
    rng = random.Random(seed ^ 0xB0A4D)
    settlements = defaultdict(list)
    roads = defaultdict(list)
    board_map = game.state.board.map

    def choose(_player, _game, actions):
        if actions[0].action_type == ActionType.BUILD_SETTLEMENT:
            weights = []
            for action in actions:
                tiles = board_map.adjacent_tiles[action.value]
                resources = {tile.resource for tile in tiles if tile.resource}
                weight = 1 + sum(map(pips, tiles)) + len(resources)
                weights.append(weight ** 3)
            return rng.choices(actions, weights=weights, k=1)[0]
        if actions[0].action_type != ActionType.BUILD_ROAD:
            raise RuntimeError("Expected an opening settlement or road")
        return rng.choice(actions)

    for _ in range(4 * player_count):
        action = game.play_tick(decide_fn=choose).action
        collection = settlements if action.action_type == ActionType.BUILD_SETTLEMENT else roads
        collection[action.color].append(action.value)
    if game.state.is_initial_build_phase or {a.action_type for a in game.playable_actions} != {ActionType.ROLL}:
        raise RuntimeError("Snapshot must be immediately before the first roll")
    if any(len(settlements[c]) != 2 or len(roads[c]) != 2 for c in game.state.colors):
        raise RuntimeError("Each player must have two opening settlements and roads")
    return settlements, roads


def snapshot(game, color, nodes, roads):
    board = game.state.board
    board_map = board.map
    tile_groups = [board_map.adjacent_tiles[node] for node in nodes]
    tiles = [tile for group in tile_groups for tile in group if tile.resource]
    production = Counter()
    for tile in tiles:
        production[tile.resource] += pips(tile)
    total_pips = sum(production.values())
    port_resources = {resource for resource, port_nodes in board_map.port_nodes.items()
                      if any(node in port_nodes for node in nodes)}
    port_fit = max((production[r] for r in port_resources if r is not None), default=0)
    # Legal settlement sites if exactly one additional road were available.
    all_edges = {tuple(sorted(edge)) for tile in board_map.land_tiles.values()
                 for edge in tile.edges.values()}
    occupied_edges = {tuple(sorted(edge)) for edge in board.roads}
    endpoints = {node for edge in roads for node in edge if node not in nodes}
    legal_sites = set(board.buildable_node_ids(color, initial_build_phase=True))
    expansion = {end for a, b in all_edges - occupied_edges
                 for start, end in ((a, b), (b, a))
                 if start in endpoints and end in legal_sites}
    expansion_pips = max((sum(map(pips, board_map.adjacent_tiles[n])) for n in expansion),
                         default=0)
    road_pips = min(production["WOOD"], production["BRICK"])
    city_pips = min(production["WHEAT"], production["ORE"])
    return {
        "settlement_1": nodes[0], "settlement_2": nodes[1],
        "road_1": "-".join(map(str, sorted(roads[0]))),
        "road_2": "-".join(map(str, sorted(roads[1]))),
        "total_pips": total_pips,
        "first_pips": sum(map(pips, tile_groups[0])),
        "second_pips": sum(map(pips, tile_groups[1])),
        "resource_diversity": len(production),
        "unique_roll_numbers": len({tile.number for tile in tiles}),
        "starting_cards": sum(bool(tile.resource) for tile in tile_groups[1]),
        "starting_road_cards": sum(tile.resource in ("WOOD", "BRICK") for tile in tile_groups[1]),
        "generic_port": int(None in port_resources),
        "port_fit_pips": port_fit,
        "expansion_sites": len(expansion),
        "best_expansion_pips": expansion_pips,
        "road_material_pips": road_pips,
        "city_material_pips": city_pips,
        "balanced_score": total_pips + 2 * len(production) + road_pips + city_pips,
        **{f"pips_{resource.lower()}": production[resource] for resource in RESOURCES},
    }


def run_game(task):
    player_count, tier, seed = task
    colors = list(Color)[:player_count]
    if tier == "basic":
        players = [WeightedRandomPlayer(color) for color in colors]
    else:
        params = ValueFunctionPlayer.Params(epsilon=0.05)
        players = [ValueFunctionPlayer(color, params=params) for color in colors]
    game = Game(players, seed=seed)
    settlements, roads = setup_game(game, seed, player_count)
    rows = [
        {"players": player_count, "tier": tier, "seed": seed, "seat": seat,
         **snapshot(game, color, settlements[color], roads[color])}
        for seat, color in enumerate(game.state.colors, start=1)
    ]
    winner = game.play()
    winning_seat = game.state.colors.index(winner) + 1 if winner is not None else None
    for row in rows:
        row["winner_seat"] = winning_seat
        row["won"] = int(row["seat"] == winning_seat) if winning_seat else None
        row["turns"] = game.state.num_turns
    return rows


def simulate(args):
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.csv")
    tasks = [(players, tier, seed) for players in SIZES for tier in RULES
             for seed in range(args.seed, args.seed + args.runs)]
    with partial.open("w", newline="", encoding="utf-8") as csv_file:
        if args.workers == 1:
            for i, rows in enumerate(map(run_game, tasks), start=1):
                write_batch(csv_file, rows, i, len(tasks))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for i, rows in enumerate(pool.map(run_game, tasks, chunksize=4), start=1):
                    write_batch(csv_file, rows, i, len(tasks))
    partial.replace(output)


def write_batch(csv_file, rows, index, total):
    writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
    if index == 1:
        writer.writeheader()
    writer.writerows(rows)
    if index % 100 == 0 or index == total:
        csv_file.flush()
        print(f"{index:,}/{total:,} games simulated", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000, help="Attempts per scenario")
    parser.add_argument("--seed", type=int, default=513000, help="First game seed")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--output", type=Path, default=Path("results/openings.csv"))
    args = parser.parse_args()
    if args.runs < 1 or args.workers < 1:
        parser.error("runs and workers must be positive")
    simulate(args)

