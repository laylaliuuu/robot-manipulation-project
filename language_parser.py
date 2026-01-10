import re
from typing import List, Tuple, Optional

class CommandParser:
    """
    Rule-based parser for natural language commands.
    Outputs object keys that match your object_map, e.g. "red_block".
    """

    def __init__(self):
        # NOTE: We assume objects are two words like "red block", "green cube"
        self.pick_place_pattern = (
            r"(?:put|place|move|pick up).*?(?:the\s+)?(\w+\s+\w+)"
            r".*?(?:on|onto|under|above).*?(?:the\s+)?(\w+\s+\w+)"
        )
        self.pick_pattern = r"(?:pick up|grab|get).*?(?:the\s+)?(\w+\s+\w+)"
        self.place_pattern = r"(?:place|put).*?(?:the\s+)?(\w+\s+\w+)"
        self.open_gripper_pattern = r"(?:open|release).*?gripper"
        self.close_gripper_pattern = r"(?:close|grip|grasp).*?gripper"
        self.stack_pattern = (
            r"(?:stack|put).*?(?:the\s+)?(\w+\s+\w+)"
            r".*?(?:on|under|over).*?(?:the\s+)?(\w+\s+\w+)"
        )

    def _norm(self, name: str) -> str:
        # "red block" -> "red_block"
        return name.strip().lower().replace(" ", "_")

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
        Normalize command with synonyms, but DO NOT change cube<->block etc.
        (because that breaks your object_map keys).
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
