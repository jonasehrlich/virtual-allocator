import unittest

import virtual_allocator.allocator as allocator


class TestAllocator(unittest.TestCase):
    def test_first_fit_allocation(self):
        alloc = allocator.Allocator(address=0, size=100, allocation_policy=allocator.AllocationPolicy.FIRST_FIT)
        for _ in range(9):
            alloc.allocate(10)

        self.assertEqual(alloc.regions, 10)

        with self.assertRaises(allocator.AllocatorOutOfMemoryError):
            alloc.allocate(20)

        alloc.allocate(10)

        self.assertEqual(alloc.regions, 10)
        self.assertEqual({region.is_free for region in alloc.regions}, {False})

    def test_free(self):
        alloc = allocator.Allocator(address=0, size=100, allocation_policy=allocator.AllocationPolicy.FIRST_FIT)
        alloc.allocate(10)
        free_region = alloc.regions[-1]
        big_region = alloc.allocate(50)
        with self.assertRaises(allocator.AllocatorOutOfMemoryError):
            alloc.allocate(50)
        alloc.free(big_region)
        self.assertEqual(alloc.regions[-1], free_region)
        # Check that the region is really unknown
        with self.assertRaises(allocator.UnknownRegionError):
            alloc._get_region_idx(big_region)
        # Check that we can allocate 50 bytes again
        alloc.allocate(50)

    def test_best_fit_allocation(self):
        """Test the best fit allocation scheme"""
        alloc = allocator.Allocator(address=0, size=100, allocation_policy=allocator.AllocationPolicy.BEST_FIT)

        regions_to_free = list()
        alloc.allocate(10)
        regions_to_free.append(alloc.allocate(20))
        alloc.allocate(10)
        regions_to_free.append(alloc.allocate(5))
        alloc.allocate(10)

        for region in regions_to_free:
            alloc.free(region)

        alloc.allocate(5)
        expected_regions = [
            allocator.MemoryRegion(allocator.Address(0), allocator.Size(10), is_free=False),
            allocator.MemoryRegion(allocator.Address(10), allocator.Size(20), is_free=True),
            allocator.MemoryRegion(allocator.Address(30), allocator.Size(10), is_free=False),
            allocator.MemoryRegion(allocator.Address(40), allocator.Size(5), is_free=False),
            allocator.MemoryRegion(allocator.Address(45), allocator.Size(10), is_free=False),
            allocator.MemoryRegion(allocator.Address(55), allocator.Size(45), is_free=True),
        ]
        self.assertEqual(alloc.regions, expected_regions)
