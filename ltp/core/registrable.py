#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

from collections import defaultdict
import importlib
from warnings import warn
from typing import TypeVar, Type, Dict, List, Optional, Iterable, Tuple, Callable, Any
import inspect

from .exceptions import RegistrationError

T = TypeVar("T")
HookType = Callable[[Type[T], str], None]


class Registrable(type):
    """可注册元类

    给基类提供以下功能

    1. 通过名字访问子类
    2. 列出子类的类型
    3. 构造生成子类对象

    """

    def __init__(cls, name, bases, namespace, alias: str = None):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, '_registry'):
            cls._registry: Dict[str, Type] = defaultdict()
            cls._hooks: Optional[List[HookType]] = None
            cls.register = classmethod(register)
            cls.weak_register = classmethod(weak_register)
            cls.hook = classmethod(hook)
            cls.by_name = classmethod(by_name)
            cls.list_available = classmethod(list_available)
            cls.is_registered = classmethod(is_registered)
            cls.iter_registered = classmethod(iter_registered)
            cls.from_params = classmethod(from_params)
            cls.default_implementation = None
        elif alias:
            cls._registry[alias] = cls
        else:
            cls._registry[name] = cls

    # Optional
    @classmethod
    def __prepare__(cls, name, bases, *, alias=False):
        return super().__prepare__(name, bases)

    # Required
    def __new__(cls, name, bases, ns, *, alias=False):
        return super().__new__(cls, name, bases, ns)


def register(
        cls: Type[T],
        name: str,
        override: bool = False,
        hooks: Optional[List[HookType]] = None,
):
    """装饰器 Class decorator for registering a subclass.

    Args:
        name: 注册名
        override (bool): 当name已经注册时，是否进行覆盖
        hooks (List[HookType]): 在注册时会被执行的Hook函数

    Raises:
        RegistrationError: 如果 override 为 false 并且 name 已经被注册
    """
    registry = cls._registry
    default_hooks = cls._hooks or []  # type: ignore

    def add_subclass_to_registry(subclass: Type[T]):
        if not inspect.isclass(subclass) or not issubclass(subclass, cls):
            raise RegistrationError(
                f"Cannot register {subclass.__name__} as {name}; "
                f"{subclass.__name__} must be a subclass of {cls.__name__}"
            )
        # Add to registry.
        # If name already registered, warn if overriding or raise an error if override not allowed.
        if name in registry:
            if not override:
                raise RegistrationError(
                    f"Cannot register {subclass.__name__} as {name}; "
                    f"name already in use for {registry[name].__name__}"
                )
            else:
                warn(f"Overriding {name} in {cls.__name__} registry")
        registry[name] = subclass
        for hook in default_hooks + (hooks or []):
            hook(subclass, name)
        return subclass

    return add_subclass_to_registry


def weak_register(
        cls: Type[T],
        name: str,
        subclass: Type[T],
        override: bool = False,
        hooks: Optional[List[HookType]] = None,
) -> Type[T]:
    """用于手动对子类进行注册

    Args:
        name (str): 子类的引用名
        subclass: 子类类型
        override(bool): 当name已经注册时，是否进行覆盖
        hooks: 在注册时会被执行的Hook函数

    Raises:
        RegistrationError: 如果 override 为 false 并且 name 已经被注册
    """
    registry = cls._registry
    default_hooks = cls._hooks or []  # type: ignore

    # 函数支持
    if not inspect.isclass(subclass) and not callable(subclass):
        raise RegistrationError(
            f"Cannot register {subclass.__name__} as {name}; "
        )
        # Add to registry.
        # If name already registered, warn if overriding or raise an error if override not allowed.
    if name in registry:
        if not override:
            raise RegistrationError(
                f"Cannot register {subclass.__name__} as {name}; "
                f"name already in use for {registry[name].__name__}"
            )
        else:
            warn(f"Overriding {name} in {cls.__name__} registry")
    registry[name] = subclass
    for hook in default_hooks + (hooks or []):
        hook(subclass, name)
    return cls


def hook(cls, hook: HookType):
    """
    函数装饰器，给某个类注册装饰器
    """
    if not cls._hooks:
        cls._hooks = []
    cls._hooks.append(hook)
    return hook


def by_name(cls: Type[T], name: str) -> Type[T]:
    """通过注册的名字取得实际的类型

    Args:
        name: 注册的名字

    Returns:
        class: Type[T] 使用 name 注册的子类

    Raises:
        RegistrationError: 如果 name 未被注册
    """
    if name in cls._registry:
        return cls._registry[name]
    elif "." in name:
        # This might be a fully qualified class name, so we'll try importing its "module"
        # and finding it there.
        parts = name.split(".")
        submodule = ".".join(parts[:-1])
        class_name = parts[-1]

        try:
            module = importlib.import_module(submodule)
        except ModuleNotFoundError:
            raise RegistrationError(
                f"tried to interpret {name} as a path to a class "
                f"but unable to import module {submodule}"
            )

        try:
            maybe_subclass = getattr(module, class_name)
        except AttributeError:
            raise RegistrationError(
                f"tried to interpret {name} as a path to a class "
                f"but unable to find class {class_name} in {submodule}"
            )

        if not inspect.isclass(maybe_subclass) or not issubclass(
                maybe_subclass, cls
        ):
            raise RegistrationError(
                f"tried to interpret {name} as a path to a class "
                f"but {class_name} is not a subclass of {cls.__name__}"
            )

        # Add subclass to registry and return it.
        cls._registry[name] = maybe_subclass
        return maybe_subclass
    else:
        # is not a qualified class name
        print(cls.list_available())
        raise RegistrationError(
            f"{name} is not a registered name for {cls.__name__}."
        )


def list_available(cls: Type[T]) -> List[str]:
    """
    列出所有的注册子类
    """
    keys: List[str]
    keys = list(cls._registry.keys())
    default = cls.default_implementation  # type: ignore

    if default is None:
        return keys
    if default not in keys:
        raise RegistrationError(
            f"Default implementation {default} is not registered"
        )
    return [default] + [k for k in keys if k != default and not k.startswith('_')]


def is_registered(cls: Type[T], name: str) -> bool:
    """
    如果 name 在类中已经注册，则返回 True
    """
    return name in cls._registry


def iter_registered(cls: Type[T]) -> Iterable[Tuple[str, Type[T]]]:
    """
    迭代已经注册的名字和对象
    """
    return cls._registry.items()


def from_params(cls, __config: dict, *args, extra: dict = None, **kwargs):
    """使用 config 生成对象

    Args:
        cls: 类型
        __config: 配置项，通常为字典，形如 {'class':'ClassName', 'init':{ 'arg': arg } }
        *args: 直接传入的arg
        extra: 根据需要传入的数据
        **kwargs: 其他参数

    Returns:
        根据参数生成的对象
    """
    if "class" not in __config:
        raise RegistrationError("Not available config!!!")

    # todo Generate Config Report
    # arg_names = inspect.getfullargspec(cls)

    class_ = cls.by_name(__config["class"])

    if "init" in __config:
        kwargs.update(__config['init'])  # 以配置文件为主

    if inspect.ismethod(getattr(class_, 'from_extra', None)):
        if extra is None:
            extra = {'config': __config}
        else:
            extra['config'] = __config
        kwargs.update(class_.from_extra(extra=extra))
    elif inspect.ismethod(getattr(cls, 'from_extra', None)):
        if extra is None:
            extra = {'config': __config}
        else:
            extra['config'] = __config
        kwargs.update(cls.from_extra(extra=extra, subcls=class_))

    return class_(*args, **kwargs)
