from __future__ import annotations

import dataclasses
import enum
import typing as ty

__version__ = "0.0.1"


__all__ = [
    "UnknownRegionError",
    "AllocationPolicy",
    "OutOfMemoryError",
    "Allocator",
    "AlignmentError",
    "MemoryRegion",
]


@dataclasses.dataclass(frozen=True)
class MemoryRegion:
    """Dataclass describing a memory region"""

    address: int  # Start address of the allocated memory
    size: int  # Size of the allocated memory
    is_free: bool  # Whether the memory region is free
    padding: int = 0  # Padding required to fit within the alignment

    @property
    def total_size(self) -> int:
        return self.size + self.padding


class UnknownRegionError(KeyError):
    """Raised when accessing unknown memory regions"""


class OutOfMemoryError(Exception):
    """Raised if no fitting region for the required memory size could be found"""


class AllocationPolicy(enum.Enum):
    """Policy for the allocation of new memory regions"""

    FIRST_FIT = enum.auto()  # Allocate to the first fitting memory region
    BEST_FIT = enum.auto()  # Allocate to the memory region with the smallest size difference to the allocated block


class AlignmentError(ValueError):
    """Raised if the requested memory size is not a multiple of the block size"""


class Allocator:
    """Linked-list allocator"""

    def __init__(self, address: int, size: int, block_size: int, alignment: int, allocation_policy: AllocationPolicy):
        self._address = address
        self._size = size
        self._allocation_policy = allocation_policy
        self._block_size = block_size
        self._alignment = alignment

        self._regions: list[MemoryRegion] = [MemoryRegion(self._address, size, is_free=True)]

    @property
    def regions(self) -> list[MemoryRegion]:
        """Get all regions currently in the allocator"""
        return self._regions

    def allocate(self, size: int) -> MemoryRegion:
        """Allocate memory of `size` in the range of the allocator

        :param size: Size of the memory region in bytes
        :type size: int
        :raises ValueError: Raised if an invalid size is passed in
        :raises OutOfMemoryError: Raised if no fitting free memory region could be found
        :return: Allocated memory region
        :rtype: MemoryRegion
        """

        if size < 0:
            raise ValueError(f"Invalid size {size}")

        free_region = self.find_free_memory_region(size)
        free_region_idx = self._get_region_idx(free_region)

        # Create the region with size 0 and insert it into the regions, use resize functionality to implement the
        # resizing of the regions
        allocated_region = MemoryRegion(free_region.address, size=0, is_free=False)
        self._regions.insert(free_region_idx, allocated_region)
        try:
            return self.resize(allocated_region, size)
        except AlignmentError:
            # Alignment error during resize, remove the empty allocated region from the list of regions
            self._regions.pop(free_region_idx)
            raise

    def resize(self, region: MemoryRegion, size: int) -> MemoryRegion:
        """Resize a memory region. The memory region can only be grown if no region is allocated after it.

        :param region_id: ID of the region to resize
        :type region_id: RegionID
        :param size: New size of the region
        :type size: int
        """
        if size < 0:
            raise ValueError(f"Invalid size {size}")

        if size % self._block_size != 0:
            raise AlignmentError(f"Size {size} is not a multiple of block size {self._block_size}")

        if size == region.size:
            return region
        if size > region.size:
            return self._increase_region_size(region, size)
        return self._decrease_region_size(region, size)

    def free(self, region: MemoryRegion) -> None:
        """Free a memory region

        If the surrounding memory regions are free as well, the regions will be merged

        :param region: Region to free
        :type region: MemoryRegion
        :raises UnknownRegionError: Raised if the region does not exist in the allocator
        """
        region_idx = self._get_region_idx(region)
        if region.is_free:
            return

        previous_region, next_region = self._get_surrounding_regions(region)
        if previous_region and previous_region.is_free and next_region and next_region.is_free:
            # The surrounding regions are not allocated, merge with both of them
            self._regions[region_idx - 1] = dataclasses.replace(
                previous_region, size=previous_region.size + region.total_size + next_region.size
            )
            self._regions.remove(region)
            self._regions.remove(next_region)

        elif previous_region and previous_region.is_free:
            # The previous region is not allocated, merge with previous region
            self._regions[region_idx - 1] = dataclasses.replace(
                previous_region, size=previous_region.size + region.total_size
            )
            self._regions.remove(region)

        elif next_region and next_region.is_free:
            # The next region is not allocated, merge with next region
            self._regions[region_idx] = dataclasses.replace(
                region, is_free=True, size=region.total_size + next_region.size
            )
            self._regions.remove(next_region)
        else:
            # The surrounding regions are allocated, just free this region
            self._regions[region_idx] = dataclasses.replace(region, size=region.total_size, padding=0, is_free=True)

    def _get_region_idx(self, region: MemoryRegion) -> int:
        """Get the index of a region in the current region list"""
        try:
            return self._regions.index(region)
        except ValueError:
            raise UnknownRegionError(f"Memory region {region} is unknown")

    def _get_padding(self, size: int) -> int:
        """Get the padding required for `size`"""
        leftover = size % self._alignment
        if leftover == 0:
            return leftover
        return self._alignment - leftover

    def _gen_free_regions(self, size: int) -> ty.Generator[MemoryRegion, None, None]:
        """Return a generator that yields all free memory regions with the minimum size.

        The minimum size is `size` plus the padding to align the following region according to the alignment configured
        for the allocator.

        :param size: Minimum Size of the Memory region
        :type size: int
        :return: Generator yielding all free regions
        :rtype: ty.Generator[MemoryRegion, None, None]
        :yield: Free memory region with matching the minimum size
        :rtype: Iterator[ty.Generator[MemoryRegion, None, None]]
        """

        padding = self._get_padding(size)
        total_size = size + padding
        for region in self._regions:
            if region.is_free:
                if region.total_size >= total_size:
                    yield region

    def _get_surrounding_regions(self, region: MemoryRegion) -> tuple[MemoryRegion | None, MemoryRegion | None]:
        """Get a 2-tuple of the regions surrounding a memory region.

        If the region is at the start or the end of the regions, the respective surrounding region will be set to `None`

        :param region: Region to get the surrounding regions of
        :type region: MemoryRegion
        :return: 2-tuple of (previous_region, next_region), if the region is at the start or the end of the regions,
                 the respective surrounding region will be set to `None`
        :rtype: tuple[MemoryRegion | None, MemoryRegion | None]
        """
        region_idx = self._get_region_idx(region)
        try:
            next_region = self._regions[region_idx + 1]
        except IndexError:
            next_region = None
        try:
            if region_idx == 0:
                raise IndexError()
            previous_region = self._regions[region_idx - 1]
        except IndexError:
            previous_region = None
        return previous_region, next_region

    def find_free_memory_region(self, size: int) -> MemoryRegion:
        """Find a free memory region according to the allocation policy

        :param size: Minimum size of the memory region
        :type size: int
        :raises OutOfMemoryError: Raised if no free memory region could be found
        :return: First free memory region that fits the required size
        :rtype: MemoryRegion
        """

        # Find free space
        if self._allocation_policy == AllocationPolicy.FIRST_FIT:
            return self.find_first_free_memory_region(size)
        elif self._allocation_policy == AllocationPolicy.BEST_FIT:
            return self.find_best_free_memory_region(size)

        raise ValueError(f"Invalid allocation policy: {self._allocation_policy}")  # pragma: no cover

    def find_first_free_memory_region(self, size: int) -> MemoryRegion:
        """Find the first free memory region that fits the required size

        :param size: Minimum size of the memory region
        :type size: int
        :raises OutOfMemoryError: Raised if no free memory region could be found
        :return: First free memory region that fits the required size
        :rtype: MemoryRegion
        """
        gen = self._gen_free_regions(size=size)
        try:
            free_region = next(gen)
            gen.close()
            return free_region
        except StopIteration:
            raise OutOfMemoryError(f"No memory region for size {size} found")

    def find_best_free_memory_region(self, size: int) -> MemoryRegion:
        """Find the best free memory region that fits the required size

        :param size: Minimum size of the memory region
        :type size: int
        :raises OutOfMemoryError: Raised if no fitting free memory region could be found
        :return: Best free memory region that fits the required size
        :rtype: MemoryRegion
        """

        fitting_regions = sorted(list(self._gen_free_regions(size=size)), key=lambda region: region.size)
        try:
            return fitting_regions[0]
        except IndexError:
            raise OutOfMemoryError(f"No memory region for size {size} found")

    def _increase_region_size(self, region: MemoryRegion, size: int) -> MemoryRegion:
        """Increase the size of a region

        :param region: Memory region to resize
        :type region: MemoryRegion
        :param size: New size of the region
        :type size: int
        :raises ValueError: Raised for sizes smaller than the current region size
        :raises OutOfMemoryError: Raised if the region cannot get resized
        :return: Memory region with the increased size
        :rtype: MemoryRegion
        """
        if size < region.size:
            raise ValueError(f"Cannot increase region size of {region} to {size}")  # pragma: no cover

        region_idx = self._get_region_idx(region)
        _, next_region = self._get_surrounding_regions(region)
        padding = self._get_padding(size)

        if next_region and next_region.is_free and (region.total_size + next_region.total_size) >= (size + padding):
            # We have space to resize the current region to the desired size
            resized_region = dataclasses.replace(region, size=size, padding=padding)
            self._regions[region_idx] = resized_region
            # The allocated region was inserted before the free region,
            # reduce the size of the following free region and remove it if the size is zero
            new_free_region = MemoryRegion(
                address=region.address + resized_region.total_size,
                size=region.total_size + next_region.total_size - resized_region.total_size,
                is_free=True,
            )

            if new_free_region.total_size:
                # Replace the previous region with one reduced in size
                self._regions[region_idx + 1] = new_free_region
            else:
                # The new free region would be empty, remove it from the list of regions
                self._regions.pop(region_idx + 1)
            return resized_region

        raise OutOfMemoryError(f"Cannot resize {region} to size {size}")

    def _decrease_region_size(self, region: MemoryRegion, size: int) -> MemoryRegion:
        """Decrease the size of a memory region

        :param region: Memory region to resize
        :type region: MemoryRegion
        :param size: New size of the region
        :type size: int
        :raises ValueError: Raised for sizes greater than the current region size
        :return: Memory region with the decreased size
        :rtype: MemoryRegion
        """

        if size > region.size:
            raise ValueError(f"Cannot reduce region size of {region} to {size}")  # pragma: no cover

        region_idx = self._get_region_idx(region)
        _, next_region = self._get_surrounding_regions(region)
        padding = self._get_padding(size)

        resized_region = dataclasses.replace(region, size=size, padding=padding)
        self._regions[region_idx] = resized_region
        if next_region and next_region.is_free:
            # Move next free region

            self._regions[region_idx + 1] = dataclasses.replace(
                next_region,
                address=resized_region.address + resized_region.total_size,
                size=next_region.total_size + region.total_size - resized_region.total_size,
            )

        else:
            # No next region, or next region is not free, insert a new region
            free_region_address = resized_region.address + resized_region.total_size
            if next_region:
                # There is a next region and it is already allocated, calculate the size of a free memory range to be
                # inserted
                free_region_size = next_region.address - free_region_address
            else:
                # There is no next region, we are at the end of the total memory range
                free_region_size = self._address + self._size - free_region_address
            self._regions.insert(
                region_idx + 1, MemoryRegion(address=free_region_address, size=free_region_size, is_free=True)
            )
        return resized_region
