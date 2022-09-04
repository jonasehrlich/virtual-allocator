import pytest
from virtual_allocator import *


def test_allocate():
    """Test allocation of memory"""
    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.FIRST_FIT)
    with pytest.raises(ValueError):
        alloc.allocate(-16)

    for _ in range(5):
        alloc.allocate(32)

    assert len(alloc.regions) == 6

    with pytest.raises(OutOfMemoryError):
        alloc.allocate(128)

    alloc.allocate(96)

    assert len(alloc.regions) == 6
    assert {region.is_free for region in alloc.regions} == {False}


def test_free(subtests):
    """Test free of memory"""
    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.FIRST_FIT)

    for _ in range(4):
        alloc.allocate(32)

    big_region = alloc.allocate(128)

    with pytest.raises(OutOfMemoryError):
        alloc.allocate(32)

    alloc.free(big_region)

    # Check that the region is really unknown
    with pytest.raises(UnknownRegionError):
        alloc._get_region_idx(big_region)

    # Check that we can allocate 128 bytes again
    big_region2 = alloc.allocate(128)
    alloc.free(big_region2)

    with subtests.test("no free surrounding"):
        alloc.free(MemoryRegion(32, 32, False))
        assert alloc.regions == [
            MemoryRegion(0, 32, False),
            MemoryRegion(32, 32, True),
            MemoryRegion(64, 32, False),
            MemoryRegion(96, 32, False),
            MemoryRegion(128, 128, True),
        ]

    with subtests.test("already free"):
        alloc.free(MemoryRegion(32, 32, True))

    with subtests.test("prev free"):
        alloc.free(MemoryRegion(64, 32, False))
        assert alloc.regions == [
            MemoryRegion(0, 32, False),
            MemoryRegion(32, 64, True),
            MemoryRegion(96, 32, False),
            MemoryRegion(128, 128, True),
        ]

    alloc.allocate(32)
    alloc.allocate(32)

    with subtests.test("next free"):
        alloc.free(MemoryRegion(96, 32, False))

        assert alloc.regions == [
            MemoryRegion(0, 32, False),
            MemoryRegion(32, 32, False),
            MemoryRegion(64, 32, False),
            MemoryRegion(96, 160, True),
        ]

    with subtests.test("next and prev free"):
        alloc.free(MemoryRegion(32, 32, False))

        assert alloc.regions == [
            MemoryRegion(0, 32, False),
            MemoryRegion(32, 32, True),
            MemoryRegion(64, 32, False),
            MemoryRegion(96, 160, True),
        ]
        alloc.free(MemoryRegion(64, 32, False))
        assert alloc.regions == [
            MemoryRegion(0, 32, False),
            MemoryRegion(32, 224, True),
        ]


def test_best_fit_allocation():
    """Test the best fit allocation policy"""
    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.BEST_FIT)

    regions_to_free = list()
    alloc.allocate(32)
    regions_to_free.append(alloc.allocate(64))
    alloc.allocate(32)
    regions_to_free.append(alloc.allocate(16))
    alloc.allocate(32)

    for region in regions_to_free:
        alloc.free(region)

    alloc.allocate(16)
    expected_regions = [
        MemoryRegion(0, 32, is_free=False),
        MemoryRegion(32, 64, is_free=True),
        MemoryRegion(96, 32, is_free=False),
        MemoryRegion(128, 16, is_free=False, padding=16),
        MemoryRegion(160, 32, is_free=False),
        MemoryRegion(192, 64, is_free=True),
    ]
    assert alloc.regions == expected_regions

    with pytest.raises(OutOfMemoryError):
        alloc.allocate(96)


def test_first_fit_allocation():
    """Test the first fit allocation policy"""
    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.FIRST_FIT)

    regions_to_free = list()
    alloc.allocate(32)
    regions_to_free.append(alloc.allocate(64))
    alloc.allocate(32)
    regions_to_free.append(alloc.allocate(16))
    alloc.allocate(32)

    for region in regions_to_free:
        alloc.free(region)

    alloc.allocate(16)
    expected_regions = [
        MemoryRegion(0, 32, is_free=False),
        MemoryRegion(32, 16, is_free=False, padding=16),
        MemoryRegion(64, 32, is_free=True),
        MemoryRegion(96, 32, is_free=False),
        MemoryRegion(128, 32, is_free=True),
        MemoryRegion(160, 32, is_free=False),
        MemoryRegion(192, 64, is_free=True),
    ]
    assert alloc.regions == expected_regions


def test_resize_decrease():
    """Test region resize when the size decreases"""

    alloc = Allocator(address=0, size=256, block_size=8, alignment=16, allocation_policy=AllocationPolicy.BEST_FIT)
    r1 = alloc.allocate(32)
    r2 = alloc.allocate(32)

    assert alloc.regions == [
        MemoryRegion(0, 32, is_free=False),
        MemoryRegion(32, 32, is_free=False),
        MemoryRegion(64, 192, is_free=True),
    ]

    alloc.resize(r2, 8)
    assert alloc.regions == [
        MemoryRegion(0, 32, is_free=False),
        MemoryRegion(32, 8, is_free=False, padding=8),
        MemoryRegion(48, 208, is_free=True),
    ]

    alloc.resize(r1, 16)
    assert alloc.regions == [
        MemoryRegion(0, 16, is_free=False),
        MemoryRegion(16, 16, is_free=True),
        MemoryRegion(32, 8, is_free=False, padding=8),
        MemoryRegion(48, 208, is_free=True),
    ]

    with pytest.raises(ValueError):
        alloc.resize(MemoryRegion(32, 8, is_free=False), -5)

    r3 = alloc.allocate(208)
    alloc.resize(r3, 192)

    assert alloc.regions == [
        MemoryRegion(0, 16, is_free=False),
        MemoryRegion(16, 16, is_free=True),
        MemoryRegion(32, 8, is_free=False, padding=8),
        MemoryRegion(48, 192, is_free=False),
        MemoryRegion(240, 16, is_free=True),
    ]


def test_resize_increase():
    """Test region resize when the size increases"""

    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.BEST_FIT)
    r1 = alloc.allocate(64)
    r2 = alloc.allocate(32)

    assert alloc.regions == [
        MemoryRegion(0, 64, is_free=False),
        MemoryRegion(64, 32, is_free=False),
        MemoryRegion(96, 160, is_free=True),
    ]

    with pytest.raises(OutOfMemoryError):
        alloc.resize(r1, r1.size + 16)

    with pytest.raises(OutOfMemoryError):
        alloc.resize(r2, 208)


def test_resize_same_size():
    """Test region resize when the size stays the same"""

    alloc = Allocator(address=0, size=256, block_size=16, alignment=32, allocation_policy=AllocationPolicy.BEST_FIT)
    alloc.allocate(64)
    r = alloc.allocate(32)
    assert alloc.regions == [
        MemoryRegion(0, 64, is_free=False),
        MemoryRegion(64, 32, is_free=False),
        MemoryRegion(96, 160, is_free=True),
    ]
    alloc.resize(r, r.size)
    assert alloc.regions == [
        MemoryRegion(0, 64, is_free=False),
        MemoryRegion(64, 32, is_free=False),
        MemoryRegion(96, 160, is_free=True),
    ]


def test_block_size():
    """Test that the block size is observed"""
    alloc = Allocator(address=0, size=128, block_size=16, alignment=1, allocation_policy=AllocationPolicy.BEST_FIT)
    with pytest.raises(AlignmentError):

        alloc.allocate(20)
    r = alloc.allocate(32)
    assert alloc.regions == [
        MemoryRegion(0, 32, is_free=False),
        MemoryRegion(32, 96, is_free=True),
    ]

    with pytest.raises(AlignmentError):
        alloc.resize(r, 35)
    alloc.resize(r, 64)
    assert alloc.regions == [
        MemoryRegion(0, 64, is_free=False),
        MemoryRegion(64, 64, is_free=True),
    ]
