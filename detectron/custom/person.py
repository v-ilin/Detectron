from enum import IntEnum


class person_class_index(IntEnum):
    in_hardhat = 1
    without_hardhat = 4
    in_gloves = 3
    without_gloves = 7
    in_goggles = 5
    without_goggles = 6
    in_belt = 9
    without_belt = 8
    in_hood = 2


class person_hardhat_status(IntEnum):
    Undefined = 0
    In_Hardhat = 1
    Without_Hardhat = 2
    In_Hood = 3


class person:
    def __init__(self, bbox, hardhat_status, gloves_on, goggles_on, in_danger_zone, belt_on):
        self.bbox = bbox
        self.hardhat_status = hardhat_status
        self.gloves_on = gloves_on
        self.goggles_on = goggles_on
        self.in_danger_zone = in_danger_zone
        self.belt_on = belt_on


class hardhat_polygon:
    status = person_hardhat_status.Undefined

    def __init__(self, bbox, status):
        self.bbox = bbox
        self.status = status


class gloves_goggles_polygon:
    status = None

    def __init__(self, bbox, status):
        self.bbox = bbox
        self.status = status
