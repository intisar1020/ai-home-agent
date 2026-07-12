"""
Pydantic schemas for validating tool arguments.
"""

from pydantic import BaseModel, Field


class DetectPersonArgs(BaseModel):
    camera: str = Field(description="Camera to search, e.g. front door, backyard, driveway, hallway, garage")
    description: str = Field(description="Appearance or attributes of the person, e.g. man in red jacket, person with backpack")


class DetectMotionArgs(BaseModel):
    camera: str = Field(description="Camera to check for motion, e.g. backyard, side gate, basement, driveway")


class TrackPersonArgs(BaseModel):
    description: str = Field(description="Description of the person to track across all cameras, e.g. person in dark hoodie, child with yellow backpack")


class CheckEntryArgs(BaseModel):
    entry: str = Field(description="Door, window, or gate to check, e.g. front door, kitchen window, side gate, patio door")


class FindEventArgs(BaseModel):
    event: str = Field(description="Security event to locate, e.g. someone entered the backyard, the delivery truck arrived, the garage door opened")


class DescribeEventArgs(BaseModel):
    timestamp: str = Field(description="Timestamp to describe, e.g. 10:45 PM, 03:15:22")


class CountPeopleArgs(BaseModel):
    camera: str = Field(description="Camera to count people in, e.g. living room, front yard, driveway, garage")


class SummarizeActivityArgs(BaseModel):
    camera: str = Field(description="Camera to summarize recent activity for, e.g. front door, backyard, driveway, basement")


class DetectVehicleArgs(BaseModel):
    camera: str = Field(description="Camera to check for vehicles, e.g. driveway, street, front yard, garage")


class ListCamerasArgs(BaseModel):
    """No arguments required."""
    pass
