from dataclasses import dataclass
from typing import List


@dataclass
class FilterItem:
    name: str
    instance: object
    enabled: bool = True


class FilterPipeline:

    def __init__(self):
        self.items: List[FilterItem] = []

    def add(self, name, instance, enabled=True):

        self.items.append(
            FilterItem(
                name=name,
                instance=instance,
                enabled=enabled
            )
        )

    def set_enabled(self, name, enabled):

        for item in self.items:

            if item.name == name:
                item.enabled = enabled
                return

    def move(self, name, new_index):

        current_index = None

        for i, item in enumerate(self.items):

            if item.name == name:
                current_index = i
                break

        if current_index is None:
            return

        item = self.items.pop(current_index)

        new_index = max(
            0,
            min(new_index, len(self.items))
        )

        self.items.insert(new_index, item)

    def apply(self, frame, landmarks):

        for item in self.items:

            if item.enabled:

                frame = item.instance.apply(
                    frame,
                    landmarks
                )

        return frame