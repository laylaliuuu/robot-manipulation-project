import re

class CommandParser:
    """
    Rule-based parser for natural language commands.
    Outputs canonical object keys that match your object_map, e.g. "red_block".
    Supports common aliases like "red cube" -> "red_block".
    """

    def __init__(self):
        # Two-word object mentions like "red block", "green cube", etc.
        self.pick_place_pattern = (
            r"(?:put|place|move|pick up).*?(?:the\s+)?(\w+(?:\s+\w+)?)"
            r".*?(?:on|onto|under|above).*?(?:the\s+)?(\w+(?:\s+\w+)?)"
        )
        self.stack_pattern = (
            r"(?:stack|put).*?(?:the\s+)?(\w+(?:\s+\w+)?)"
            r".*?(?:on|under|over).*?(?:the\s+)?(\w+(?:\s+\w+)?)"
        )
        self.pick_pattern = r"(?:pick up|grab|get).*?(?:the\s+)?(\w+\s+\w+)"
        self.place_pattern = r"(?:place|put).*?(?:the\s+)?(\w+\s+\w+)"
        self.open_gripper_pattern = r"(?:open|release).*?gripper"
        self.close_gripper_pattern = r"(?:close|grip|grasp).*?gripper"

        # Canonical object names in your sim
        self.alias = {
            # canonical
            "red block": "red_block",
            "green cube": "green_cube",
            "blue sphere": "blue_sphere",
            "white cube": "white_cube",
            "table": "the_table",
            "the table": "the_table",

            # common user slips
            "red cube": "red_block",
            "green block": "green_cube",
            "blue cube": "blue_sphere",   # you only have a blue_sphere right now
            "white block": "white_cube",

            # optional single-word shortcuts (comment out if you dislike)
            "red": "red_block",
            "green": "green_cube",
            "blue": "blue_sphere",
            "white": "white_cube",
        }

    def _norm(self, name: str) -> str:
        """
        Normalize user object mention to canonical object_map key.
        """
        raw = name.strip().lower()
        raw = re.sub(r"\s+", " ", raw)  # collapse multiple spaces

        if raw in self.alias:
            return self.alias[raw]

        # fallback: "red block" -> "red_block"
        return raw.replace(" ", "_")

    def parse(self, command: str):
        command = command.lower().strip()
        actions = []

        # pick-and-place
        match = re.search(self.pick_place_pattern, command)
        if match:
            obj, target = match.groups()
            actions.append(("pick", self._norm(obj), None))
            actions.append(("place", self._norm(target), None))
            return actions

        # stack
        match = re.search(self.stack_pattern, command)
        if match:
            obj, target = match.groups()
            actions.append(("pick", self._norm(obj), None))
            actions.append(("place", self._norm(target), None))
            return actions

        # pick only
        match = re.search(self.pick_pattern, command)
        if match:
            obj = match.group(1)
            actions.append(("pick", self._norm(obj), None))
            return actions

        # place only
        match = re.search(self.place_pattern, command)
        if match:
            target = match.group(1)
            actions.append(("place", self._norm(target), None))
            return actions

        # gripper commands
        if re.search(self.open_gripper_pattern, command):
            actions.append(("open_gripper", None, None))
            return actions

        if re.search(self.close_gripper_pattern, command):
            actions.append(("close_gripper", None, None))
            return actions

        return None

    def normalize_command(self, command: str) -> str:
        """
        Normalize command with synonyms, but do NOT change cube<->block etc.
        Aliasing is handled in _norm().
        """
        synonyms = {
            "grab": "pick up",
            "move": "place",
            "put": "place",
            "set": "place",
        }
        normalized = command.lower()
        for old, new in synonyms.items():
            normalized = re.sub(r"\b" + old + r"\b", new, normalized)
        return normalized
