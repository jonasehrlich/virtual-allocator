import pytest
from virtual_allocator import allocator


def test_allocate():
    """Test allocation of memory"""
    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.FIRST_FIT
    )
    for _ in range(9):
        alloc.allocate(10)

    assert len(alloc.regions) == 10

    with pytest.raises(allocator.AllocatorOutOfMemoryError):
        alloc.allocate(20)

    alloc.allocate(10)

    assert len(alloc.regions) == 10
    assert {region.is_free for region in alloc.regions} == {False}


def test_free(subtests):
    """Test free of memory"""
    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.FIRST_FIT
    )

    for _ in range(4):
        alloc.allocate(10)

    big_region = alloc.allocate(50)

    with pytest.raises(allocator.AllocatorOutOfMemoryError):
        alloc.allocate(30)

    alloc.free(big_region)

    # Check that the region is really unknown
    with pytest.raises(allocator.UnknownRegionError):
        alloc._get_region_idx(big_region)
    # Check that we can allocate 50 bytes again
    big_region2 = alloc.allocate(50)
    alloc.free(big_region2)

    with subtests.test("no free surrounding"):
        alloc.free(allocator.MemoryRegion(10, 10, False))
        assert alloc.regions == [
            allocator.MemoryRegion(0, 10, False),
            allocator.MemoryRegion(10, 10, True),
            allocator.MemoryRegion(20, 10, False),
            allocator.MemoryRegion(30, 10, False),
            allocator.MemoryRegion(40, 60, True),
        ]

    with subtests.test("already free"):
        alloc.free(allocator.MemoryRegion(10, 10, True))

    with subtests.test("prev free"):
        alloc.free(allocator.MemoryRegion(20, 10, False))
        assert alloc.regions == [
            allocator.MemoryRegion(0, 10, False),
            allocator.MemoryRegion(10, 20, True),
            allocator.MemoryRegion(30, 10, False),
            allocator.MemoryRegion(40, 60, True),
        ]

    alloc.allocate(10)
    alloc.allocate(10)

    with subtests.test("next free"):
        alloc.free(allocator.MemoryRegion(30, 10, False))

        assert alloc.regions == [
            allocator.MemoryRegion(0, 10, False),
            allocator.MemoryRegion(10, 10, False),
            allocator.MemoryRegion(20, 10, False),
            allocator.MemoryRegion(30, 70, True),
        ]

    with subtests.test("next and prev free"):
        alloc.free(allocator.MemoryRegion(10, 10, False))

        assert alloc.regions == [
            allocator.MemoryRegion(0, 10, False),
            allocator.MemoryRegion(10, 10, True),
            allocator.MemoryRegion(20, 10, False),
            allocator.MemoryRegion(30, 70, True),
        ]
        alloc.free(allocator.MemoryRegion(20, 10, False))
        assert alloc.regions == [
            allocator.MemoryRegion(0, 10, False),
            allocator.MemoryRegion(10, 90, True),
        ]


def test_best_fit_allocation():
    """Test the best fit allocation policy"""
    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )

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
        allocator.MemoryRegion(0, 10, is_free=False),
        allocator.MemoryRegion(10, 20, is_free=True),
        allocator.MemoryRegion(30, 10, is_free=False),
        allocator.MemoryRegion(40, 5, is_free=False),
        allocator.MemoryRegion(45, 10, is_free=False),
        allocator.MemoryRegion(55, 45, is_free=True),
    ]
    assert alloc.regions == expected_regions

    with pytest.raises(allocator.AllocatorOutOfMemoryError):
        alloc.allocate(60)


def test_first_fit_allocation():
    """Test the first fit allocation policy"""
    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.FIRST_FIT
    )

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
        allocator.MemoryRegion(0, 10, is_free=False),
        allocator.MemoryRegion(10, 5, is_free=False),
        allocator.MemoryRegion(15, 15, is_free=True),
        allocator.MemoryRegion(30, 10, is_free=False),
        allocator.MemoryRegion(40, 5, is_free=True),
        allocator.MemoryRegion(45, 10, is_free=False),
        allocator.MemoryRegion(55, 45, is_free=True),
    ]
    assert alloc.regions == expected_regions


def test_resize_decrease():
    """Test region resize when the size decreases"""

    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )
    r1 = alloc.allocate(10)
    r2 = alloc.allocate(10)

    assert alloc.regions == [
        allocator.MemoryRegion(0, 10, is_free=False),
        allocator.MemoryRegion(10, 10, is_free=False),
        allocator.MemoryRegion(20, 80, is_free=True),
    ]

    alloc.resize(r2, 5)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 10, is_free=False),
        allocator.MemoryRegion(10, 5, is_free=False),
        allocator.MemoryRegion(15, 85, is_free=True),
    ]

    alloc.resize(r1, 5)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 5, is_free=False),
        allocator.MemoryRegion(5, 5, is_free=True),
        allocator.MemoryRegion(10, 5, is_free=False),
        allocator.MemoryRegion(15, 85, is_free=True),
    ]

    with pytest.raises(ValueError):
        alloc.resize(allocator.MemoryRegion(10, 5, is_free=False), -5)


def test_resize_increase():
    """Test region resize when the size increases"""

    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )
    r1 = alloc.allocate(20)
    r2 = alloc.allocate(10)

    assert alloc.regions == [
        allocator.MemoryRegion(0, 20, is_free=False),
        allocator.MemoryRegion(20, 10, is_free=False),
        allocator.MemoryRegion(30, 70, is_free=True),
    ]

    with pytest.raises(allocator.AllocatorOutOfMemoryError):
        alloc.resize(r1, r1.size + 1)

    with pytest.raises(allocator.AllocatorOutOfMemoryError):
        alloc.resize(r2, 85)


def test_resize_same_size():
    """Test region resize when the size stays the same"""

    alloc = allocator.Allocator(
        address=0, size=100, block_size=1, alignment=1, allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )
    alloc.allocate(20)
    r = alloc.allocate(10)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 20, is_free=False),
        allocator.MemoryRegion(20, 10, is_free=False),
        allocator.MemoryRegion(30, 70, is_free=True),
    ]
    alloc.resize(r, r.size)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 20, is_free=False),
        allocator.MemoryRegion(20, 10, is_free=False),
        allocator.MemoryRegion(30, 70, is_free=True),
    ]


def test_block_size():
    """Test that the"""
    alloc = allocator.Allocator(
        address=0, size=128, block_size=16, alignment=1, allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )
    with pytest.raises(allocator.AlignmentError):

        alloc.allocate(20)
    r = alloc.allocate(32)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 32, is_free=False),
        allocator.MemoryRegion(32, 96, is_free=True),
    ]

    with pytest.raises(allocator.AlignmentError):
        alloc.resize(r, 35)
    alloc.resize(r, 64)
    assert alloc.regions == [
        allocator.MemoryRegion(0, 64, is_free=False),
        allocator.MemoryRegion(64, 64, is_free=True),
    ]
