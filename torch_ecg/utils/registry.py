"""
Generic registry implementation for managing and building modules (backbones, models, etc.)
"""

from typing import Any, Dict, List, Optional, Union

__all__ = [
    "Registry",
]


class Registry:
    """Registry for managing and building modules.

    A registry is used to map strings (module names) to classes, and provides
    a unified interface to instantiate modules from configurations.

    Parameters
    ----------
    name : str
        Name of the registry.

    Examples
    --------
    >>> BACKBONES = Registry("backbones")
    >>> @BACKBONES.register()
    ... class ResNet(nn.Module):
    ...     def __init__(self, depth):
    ...         self.depth = depth
    >>> # Build from string
    >>> model = BACKBONES.build("ResNet", depth=50)
    >>> # Build from config dict
    >>> model = BACKBONES.build({"name": "ResNet", "depth": 101})

    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, type] = {}

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, name: str) -> bool:
        return name in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None, override: bool = False) -> Any:
        """Decorator to register a module.

        Parameters
        ----------
        name : str, optional
            Name of the module. If not specified, the class name will be used.
        override : bool, default False
            Whether to override the existing module with the same name.

        """

        def _register(cls: type) -> type:
            _name = name or cls.__name__
            if not override and _name in self._module_dict:
                raise KeyError(f"{_name} is already registered in {self._name} registry")
            self._module_dict[_name] = cls
            return cls

        return _register

    def get(self, name: str) -> Optional[type]:
        """Get the module class by name.

        Parameters
        ----------
        name : str
            Name of the module.

        Returns
        -------
        type or None
            The registered module class.

        """
        return self._module_dict.get(name)

    def list_all(self) -> List[str]:
        """List all registered modules.

        Returns
        -------
        List[str]
            A list of all registered module names.

        """
        return list(self._module_dict.keys())

    def build(self, config: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
        """Build a module from a configuration.

        Parameters
        ----------
        config : str or dict
            Configuration of the module.
            If it's a string, it should be the name of the registered module.
            If it's a dict, it must contain a "name" key.
        **kwargs : Any
            Additional keyword arguments passed to the module's constructor.

        Returns
        -------
        Any
            The instantiated module.

        """
        if isinstance(config, str):
            obj_type = config
            obj_config = {}
        elif isinstance(config, dict):
            obj_config = config.copy()
            if "name" not in obj_config:
                raise KeyError(f"Configuration for {self._name} must contain a 'name' key")
            obj_type = obj_config.pop("name")
        else:
            raise TypeError(f"Config must be a str or dict, but got {type(config)}")

        obj_cls = self.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not registered in the {self._name} registry")

        # Merge config from dict and extra kwargs
        final_kwargs = {**obj_config, **kwargs}
        return obj_cls(**final_kwargs)
