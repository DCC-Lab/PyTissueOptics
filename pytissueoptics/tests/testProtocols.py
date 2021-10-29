from typing import Protocol, TypeVar, Sized, Iterable
# from typing import TypeVar, Reversible, Iterable, Sized

T = TypeVar('T')

class ListLike(Sized, Protocol[T]):
    def append(self, x: T) -> None:
        pass

def populate(lst: ListLike[int]) -> None:
    self.container.extend(lst)

class Scalars(Sized, Iterable[T]):
    def __init__(self, T):
        self.container: [T] = []

    def __iter__(self):
        return self.container.__iter__()

    def __len__(self):
        return self.container.__len__()

    def append(self, item):
        self.container.append(item)

groups = Scalars(int)
groups.append(1)
groups.append(2)
groups.append('text')

print(groups.container)
for item in groups:
    print(item)


