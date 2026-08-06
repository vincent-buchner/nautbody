from collections.abc import Awaitable
from typing import Protocol, TypeVar

IRequest_contra = TypeVar("IRequest_contra", contravariant=True)
IResponse_co = TypeVar("IResponse_co", covariant=True)


class UseCase(Protocol[IRequest_contra, IResponse_co]):
    def execute(
        self, request: IRequest_contra | None = None
    ) -> Awaitable[IResponse_co] | IResponse_co: ...
