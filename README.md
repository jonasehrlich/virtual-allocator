# virtual-allocator

Python allocator for a virtual memory range. This package only implements the allocation of a memory range,
the actual memory access must be implemented separately. So no use-after-free errors can be handled.

The allocator implements the `allocate`, `free` and `resize` methods which each return a new `MemoryRegion` object.

``` python


    alloc = allocator.Allocator(
        address=0,
        size=256,
        block_size=16,
        alignment=32,
        allocation_policy=allocator.AllocationPolicy.BEST_FIT
    )

    mem_ranges = [alloc.allocate(64) for _ in range(3)]

    assert mem_ranges == [
        allocator.MemoryRange(address=0, size=64, is_free=False, padding=0),
        allocator.MemoryRange(address=64, size=64, is_free=False, padding=0),
        allocator.MemoryRange(address=128, size=64, is_free=False, padding=0),
        allocator.MemoryRange(address=196, size=64, is_free=True, padding=0),
    ]

    alloc.free(mem_ranges[1])
    
    assert mem_ranges == [
        allocator.MemoryRange(address=0, size=64, is_free=False, padding=0),
        allocator.MemoryRange(address=64, size=64, is_free=True, padding=0),
        allocator.MemoryRange(address=128, size=64, is_free=False, padding=0),
        allocator.MemoryRange(address=196, size=64, is_free=True, padding=0),
    ]
```

## Allocation policies

The allocator class supports two allocation policies, `FIRST_FIT` and `BEST_FIT`.

* `FIRST_FIT` allocation allocates new regions into the lowest free region
* `BEST_FIT` allocation will allocate new regions into the free region which will create the smallest leftover memory
  range
