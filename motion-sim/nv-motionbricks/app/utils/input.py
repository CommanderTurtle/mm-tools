# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keyboard and controller input shared by the apps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Protocol

import numpy as np

from ..runtime.common.motion import CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE, LINEAR_LOCOMOTION_MODE

DEFAULT_STICK_DEAD_ZONE = 0.20
_ZERO_AXES = (0.0, 0.0)
# Each key and its controller button switch between two modes.
_MODE_TOGGLES = (
    ("o", "y", LINEAR_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE),
    ("p", "b", CROUCH_LOCOMOTION_MODE, DIRECTIONAL_LOCOMOTION_MODE),
)
_STYLE_KEY, _STYLE_BUTTON = "[", "a"


def camera_relative_movement(
    camera_facing: np.ndarray, movement: tuple[float, float],
) -> np.ndarray:
    """Map (right, forward) movement onto the ground while retaining stick strength.

    A stick at rest gives a zero vector, which MovementControl reads as no movement.
    """
    right, forward = movement
    camera_right = np.array([camera_facing[1], -camera_facing[0], 0.0], dtype=np.float32)
    return camera_facing * forward + camera_right * right


class _Controller(Protocol):
    leftx: float
    lefty: float
    rightx: float
    righty: float
    dpadx: float
    dpady: float
    a: bool
    x: bool
    y: bool
    b: bool

    def open(self) -> None: ...

    def close(self) -> None: ...


class _ControllerManager(Protocol):
    def get_controllers(self) -> list[_Controller]: ...

    def push_handlers(self, *args, **kwargs) -> None: ...

    def remove_handlers(self, *args, **kwargs) -> None: ...


@dataclass(frozen=True, slots=True)
class InputFrame:
    """Movement, camera, and speed input for this frame.

    Movement uses (right, forward) and camera orbit uses (right, up). Each
    vector's length is the stick strength after the dead zone.
    """

    movement: tuple[float, float]
    camera_orbit: tuple[float, float]
    use_max_speed: bool


def apply_radial_dead_zone(x: float, y: float, dead_zone: float) -> tuple[float, float]:
    """Ignore small stick movements and scale the remaining range from zero to one."""

    if not 0.0 <= dead_zone < 1.0:
        raise ValueError("dead_zone must be in the range [0, 1).")

    length = math.hypot(x, y)
    if length <= dead_zone:
        return _ZERO_AXES

    direction_x = x / length
    direction_y = y / length
    magnitude = (min(length, 1.0) - dead_zone) / (1.0 - dead_zone)
    return direction_x * magnitude, direction_y * magnitude


class InputManager:
    """Read the keyboard and use the first available controller."""

    def __init__(
        self,
        is_key_down: Callable[[str], bool],
        controller_manager: _ControllerManager | None = None,
        movement_dead_zone: float = DEFAULT_STICK_DEAD_ZONE,
        camera_dead_zone: float = DEFAULT_STICK_DEAD_ZONE,
    ) -> None:
        """Read keys through is_key_down, and a controller through controller_manager.

        The dead zones apply to the movement and camera sticks.
        """
        self._is_key_down = is_key_down
        self._controller_manager = controller_manager
        self._controller: _Controller | None = None
        self.movement_dead_zone = movement_dead_zone
        self.camera_dead_zone = camera_dead_zone

        if controller_manager is not None:
            controller_manager.push_handlers(self)
            self._open_first_available_controller()

    @classmethod
    def from_pyglet(cls, is_key_down: Callable[[str | int], bool]) -> InputManager:
        """Set up keyboard input and optional controller support.

        The viewer reads letters, digits, and arrow keys by name. [ goes through its pyglet
        key code.
        """

        from pyglet.window import key

        key_codes = {"[": key.BRACKETLEFT}
        try:
            from pyglet.input import ControllerManager

            controller_manager = ControllerManager()
        except Exception:
            controller_manager = None
        return cls(lambda name: is_key_down(key_codes.get(name, name)), controller_manager)

    def is_key_down(self, key: str) -> bool:
        """Return whether a keyboard key is currently held."""

        return self._is_key_down(key)

    def is_interaction_down(self) -> bool:
        """Return whether I or the controller's X button is held."""

        return self.is_key_down("i") or self.is_controller_button_down("x")

    def is_controller_button_down(self, button: Literal["a", "x", "y", "b"]) -> bool:
        """Return whether one of the demo's controller buttons is held."""

        controller = self._controller
        return controller is not None and bool(
            {"a": controller.a, "x": controller.x, "y": controller.y, "b": controller.b}[button]
        )

    def poll(self) -> InputFrame:
        """Sample movement, camera, and speed input for one update."""

        controller_movement = self._controller_movement()
        if controller_movement == _ZERO_AXES:
            # W, A, S, and D or the arrow keys give a full strength direction, so diagonals
            # are not faster.
            right = self.is_key_down("d") or self.is_key_down("right")
            left = self.is_key_down("a") or self.is_key_down("left")
            forward = self.is_key_down("w") or self.is_key_down("up")
            backward = self.is_key_down("s") or self.is_key_down("down")
            movement = apply_radial_dead_zone(float(right) - float(left), float(forward) - float(backward), 0.0)
        else:
            movement = controller_movement

        return InputFrame(
            movement=movement,
            camera_orbit=self._controller_camera_orbit(),
            use_max_speed=(
                controller_movement != _ZERO_AXES
                or self.is_key_down("space")
            ),
        )

    def close(self) -> None:
        """Stop listening for controller input and connection changes."""

        if self._controller_manager is not None:
            self._controller_manager.remove_handlers(self)
        self._close_controller()

    def on_connect(self, controller: _Controller) -> None:
        """Use a newly connected controller when no controller is active."""

        if self._controller is None:
            self._open_controller(controller)

    def on_disconnect(self, controller: _Controller) -> None:
        """Drop a disconnected controller and select another if available."""

        if controller is not self._controller:
            return
        self._controller = None
        self._open_first_available_controller()

    def _open_first_available_controller(self) -> None:
        try:
            controllers = self._controller_manager.get_controllers()
        except Exception:
            return
        for controller in controllers:
            if self._open_controller(controller):
                return

    def _open_controller(self, controller: _Controller) -> bool:
        try:
            controller.open()
        except Exception:
            return False
        self._controller = controller
        return True

    def _close_controller(self) -> None:
        controller = self._controller
        self._controller = None
        if controller is None:
            return
        try:
            controller.close()
        except Exception:
            pass

    def _controller_movement(self) -> tuple[float, float]:
        controller = self._controller
        if controller is None:
            return _ZERO_AXES

        movement = apply_radial_dead_zone(
            float(controller.leftx),
            float(controller.lefty),
            self.movement_dead_zone,
        )
        if movement != _ZERO_AXES:
            return movement

        # The controller's D-pad can also move the character.
        return apply_radial_dead_zone(
            float(controller.dpadx),
            float(controller.dpady),
            self.movement_dead_zone,
        )

    def _controller_camera_orbit(self) -> tuple[float, float]:
        controller = self._controller
        if controller is None:
            return _ZERO_AXES
        return apply_radial_dead_zone(
            float(controller.rightx),
            float(controller.righty),
            self.camera_dead_zone,
        )


class LocomotionModeInput:
    """Change the locomotion mode once per key or button press.

    O (Y on a controller) switches between Forward and Strafing. P (B on a
    controller) switches between Crouching and Strafing. [ (A on a controller)
    cycles through the style modes. From any other mode, it returns to the last
    style used.
    """

    def __init__(self, style_modes: tuple[str, ...] = ()) -> None:
        """Track the toggles. style_modes names the style modes in the order [ cycles them."""
        self._style_modes = style_modes
        self._last_style = style_modes[0] if style_modes else None
        self._held = dict.fromkeys([key for key, *_ in _MODE_TOGGLES] + [_STYLE_KEY], False)

    def poll(self, input_manager: InputManager, current_mode: str) -> str | None:
        """Return the mode that a newly pressed toggle asks for, or None."""
        if current_mode in self._style_modes:
            self._last_style = current_mode
        requested_mode = None
        for key, button, first_mode, second_mode in _MODE_TOGGLES:
            if self._pressed(input_manager, key, button):
                requested_mode = second_mode if current_mode == first_mode else first_mode
        if self._pressed(input_manager, _STYLE_KEY, _STYLE_BUTTON) and self._style_modes:
            requested_mode = self._next_style(current_mode)
        return requested_mode

    def _pressed(self, input_manager: InputManager, key: str, button: str) -> bool:
        """Return whether the key or its button went down since the last poll."""
        down = input_manager.is_key_down(key) or input_manager.is_controller_button_down(button)
        pressed = down and not self._held[key]
        self._held[key] = down
        return pressed

    def _next_style(self, current_mode: str) -> str:
        if current_mode not in self._style_modes:
            return self._last_style
        index = self._style_modes.index(current_mode)
        return self._style_modes[(index + 1) % len(self._style_modes)]
