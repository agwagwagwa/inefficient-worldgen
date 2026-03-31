"""
Block palette: maps the hundreds of Minecraft block types down to 8 categories.

Index 0: AIR        - air, cave_air, void_air
Index 1: STONE      - stone, cobblestone, granite, diorite, andesite, gravel, ores, etc.
Index 2: DIRT       - dirt, grass_block, coarse_dirt, farmland, podzol, mycelium
Index 3: SAND       - sand, red_sand, sandstone, clay, terracotta variants
Index 4: WATER      - water, flowing_water, ice, packed_ice, blue_ice
Index 5: WOOD       - logs, planks, leaves, all wood variants
Index 6: GREENERY   - grass, ferns, flowers, vines, crops, mushrooms, saplings
Index 7: MISC_SOLID - bedrock, obsidian, everything else that's solid

The palette is intentionally coarse -- we want the model to learn terrain *shape*
and broad material distribution, not individual block types.
"""

NUM_BLOCK_TYPES = 8

# Human-readable names for visualization
BLOCK_NAMES = [
    "air",
    "stone",
    "dirt",
    "sand",
    "water",
    "wood",
    "greenery",
    "misc_solid",
]

# Colors for visualization (RGBA, 0-1 range)
BLOCK_COLORS = [
    (0.0, 0.0, 0.0, 0.0),  # air - transparent
    (0.5, 0.5, 0.5, 1.0),  # stone - gray
    (0.45, 0.32, 0.18, 1.0),  # dirt - brown
    (0.85, 0.82, 0.55, 1.0),  # sand - tan
    (0.2, 0.4, 0.8, 0.7),  # water - blue, slightly transparent
    (0.4, 0.26, 0.13, 1.0),  # wood - dark brown
    (0.2, 0.65, 0.2, 1.0),  # greenery - green
    (0.15, 0.15, 0.15, 1.0),  # misc_solid - dark gray
]

# Chunk dimensions (Y, Z, X) -- Y is vertical, height-first
CHUNK_Y = 128
CHUNK_Z = 16
CHUNK_X = 16
CHUNK_SHAPE = (CHUNK_Y, CHUNK_Z, CHUNK_X)

# Kernel size for the neighbor grid
KERNEL_SIZE = 3  # 3x3 grid of chunks

# Mapping from Minecraft block names to our palette indices.
# This maps name substrings, checked in order. First match wins.
# The key is checked with `in`, so "stone" matches "minecraft:stone",
# "minecraft:cobblestone", "minecraft:stone_bricks", etc.
#
# Order matters: more specific patterns must come before general ones.
_PALETTE_RULES: list[tuple[str, int]] = [
    # Air variants (must be first -- cave_air, void_air)
    ("air", 0),
    # Flowing water/lava should be treated as air (transient, recalculated by game)
    ("flowing_water", 0),
    ("flowing_lava", 0),
    # Water / ice (source blocks only now)
    ("water", 4),
    ("ice", 4),
    ("bubble_column", 4),
    # Wood / tree stuff (before generic checks)
    ("log", 5),
    ("wood", 5),
    ("plank", 5),
    ("leaves", 5),
    ("stripped_", 5),
    # Greenery (before dirt, since tall_grass contains "grass")
    ("tall_grass", 6),
    ("grass", 6),  # note: grass_block will also match -- we handle below
    ("fern", 6),
    ("flower", 6),
    ("rose", 6),
    ("dandelion", 6),
    ("poppy", 6),
    ("tulip", 6),
    ("orchid", 6),
    ("allium", 6),
    ("lilac", 6),
    ("peony", 6),
    ("sunflower", 6),
    ("vine", 6),
    ("lily_pad", 6),
    ("mushroom", 6),
    ("sapling", 6),
    ("wheat", 6),
    ("carrot", 6),
    ("potato", 6),
    ("beetroot", 6),
    ("melon", 6),
    ("pumpkin", 6),
    ("sugar_cane", 6),
    ("cactus", 6),
    ("bamboo", 6),
    ("sweet_berry", 6),
    ("kelp", 6),
    ("seagrass", 6),
    ("dead_bush", 6),
    # Sand / clay / terracotta
    ("sand", 3),
    ("clay", 3),
    ("terracotta", 3),
    ("concrete", 3),
    # Dirt variants (grass_block is dirt-category -- it's terrain, not greenery)
    ("grass_block", 2),
    ("dirt", 2),
    ("podzol", 2),
    ("mycelium", 2),
    ("farmland", 2),
    ("soul_soil", 2),
    ("mud", 2),
    # Stone variants (ores, cobblestone, etc.)
    ("stone", 1),
    ("cobble", 1),
    ("granite", 1),
    ("diorite", 1),
    ("andesite", 1),
    ("ore", 1),
    ("gravel", 1),
    ("deepslate", 1),
    ("tuff", 1),
    ("calcite", 1),
    ("dripstone", 1),
    ("basalt", 1),
    ("blackstone", 1),
    ("netherrack", 1),
    # Bedrock
    ("bedrock", 7),
    ("obsidian", 7),
    # Lava (misc_solid -- it's hot rock basically)
    ("lava", 7),
]


def block_name_to_palette(block_name: str) -> int:
    """Map a Minecraft block name string to a palette index (0-7).

    Args:
        block_name: Full block name like 'minecraft:stone' or just 'stone'.

    Returns:
        Palette index 0-7.
    """
    name = block_name.lower()
    # Strip namespace
    if ":" in name:
        name = name.split(":", 1)[1]

    for pattern, idx in _PALETTE_RULES:
        if pattern in name:
            return idx

    # Anything we don't recognize: if it's probably a solid block, misc_solid.
    # This catches things like rails, redstone, etc.
    return 7
